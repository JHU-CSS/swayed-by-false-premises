import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("data") / "outputs"
TABLES = Path("tables")
FIGS = Path("figures")

TAU = 1e-6
B_FEVER = 1000
B_ABL = 2000
RNG = np.random.default_rng(0)
COMMIT = ["low", "medium", "high"]

DISPLAY = {
    "meta": "Llama 4 Scout",
    "mistral": "Mistral Large",
    "claude_sonnet": "Claude Sonnet",
    "claude_opus": "Claude Opus",
    "claude_haiku": "Claude Haiku",
    "gemma-3-4b-it": "Gemma 3 4B",
}
ABL_KEY = {
    "meta": "llama4_scout",
    "mistral": "mistral_large",
    "claude_sonnet": "claude_sonnet",
    "claude_opus": "claude_opus",
    "claude_haiku": "claude_haiku",
    "gemma-3-4b-it": "gemma_3_4b",
}
MODEL_ORDER = list(DISPLAY)


def configure(out_dir, fig_dir, table_dir):
    global OUT, FIGS, TABLES
    OUT, FIGS, TABLES = Path(out_dir), Path(fig_dir), Path(table_dir)
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def reseed(seed):
    global RNG
    RNG = np.random.default_rng(seed)


def save(fig, name):
    fig.savefig(FIGS / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def s_from_counts(n_ok, n_a, n_b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10((n_a / n_ok + TAU) / (n_b / n_ok + TAU))


def parse_yn(x):
    if not isinstance(x, str):
        return None
    t = x.strip().lower()
    return t if t in ("yes", "no") else None


def abl_norm(x, direct=False):
    if not isinstance(x, str):
        return None
    t = x.strip().lower()
    if t in ("yes", "no"):
        return t
    for pat in (r"so the answer is\W*(yes|no)", r"final answer is:?\W*(yes|no)"):
        hits = re.findall(pat, t)
        if hits:
            return hits[-1]
    if direct:
        m = re.match(r"^\W*(yes|no)\b", t)
        if m:
            return m.group(1)
    return None


def boot_idx(n, b):
    return RNG.integers(0, n, size=(b, n))


def chunked_sums(M, B, chunk=100):
    n = M.shape[0]
    out = []
    for start in range(0, B, chunk):
        idx = boot_idx(n, min(chunk, B - start))
        out.append(M[idx].sum(axis=1))
    return np.concatenate(out, axis=0)


def ci(x, axis=0):
    return np.nanpercentile(x, 2.5, axis=axis), np.nanpercentile(x, 97.5, axis=axis)


def fever_cond_matrix(df, pre):
    plus_cols = [c for c in df.columns if c.startswith(f"answer_{pre}_plus_")]
    conds, mats = [], []
    g = df["ground_truth"].astype(str).str.strip().str.lower().eq("yes").to_numpy()
    for pc in plus_cols:
        cond = pc.replace(f"answer_{pre}_plus_", "")
        mc = f"answer_{pre}_minus_{cond}"
        a = df[pc].map(parse_yn)
        b = df[mc].map(parse_yn)
        ok = (a.notna() & b.notna()).to_numpy()
        ay = a.eq("yes").to_numpy()
        by = b.eq("yes").to_numpy()
        corr = np.where(g, ay, ~by)
        wrong = np.where(g, by, ~ay)
        mats.append(np.stack([ok, ok & ay, ok & by, ok & corr, ok & wrong], axis=1).astype(float))
        conds.append(cond)
    return conds, np.stack(mats, axis=1)


def fever_group_stats(sums, conds):
    n_ok = sums[..., 0]
    s = s_from_counts(n_ok, sums[..., 1], sums[..., 2])
    rd = (sums[..., 1] - sums[..., 2]) / n_ok
    acc_c = sums[..., 3] / n_ok
    acc_w = sums[..., 4] / n_ok
    res = {}
    for lvl in COMMIT + ["all"]:
        sel = [i for i, c in enumerate(conds) if lvl == "all" or c.split("_")[0] == lvl]
        res[lvl] = (
            np.nanmean(s[..., sel], axis=-1),
            np.nanmean(acc_c[..., sel], axis=-1),
            np.nanmean(acc_w[..., sel], axis=-1),
            np.nanmean(rd[..., sel], axis=-1),
        )
    return res


def run_fever(seed=0):
    reseed(seed)
    rows = []
    for m in MODEL_ORDER:
        for setting, (kind, pre, ref_col) in {
            "plain": ("baseline", "pp", "answer_no_pp"),
            "CoT": ("cot", "cot_pp", "answer_cot_no_pp"),
        }.items():
            df = pd.read_csv(OUT / "fever" / f"{m}_{kind}_results.csv")
            ref = pd.read_csv(OUT / "fever" / f"{m}_{kind}_no_nudge_results.csv")
            conds, M = fever_cond_matrix(df, pre)
            point = fever_group_stats(M.sum(axis=0), conds)
            boot = fever_group_stats(chunked_sums(M, B_FEVER), conds)
            ra = ref[ref_col].map(parse_yn)
            rok = ra.notna()
            gt = ref["ground_truth"].astype(str).str.strip().str.lower()
            acc_no = float((ra[rok] == gt[rok]).mean())
            for lvl in COMMIT + ["all"]:
                s_lo, s_hi = ci(boot[lvl][0])
                c_lo, c_hi = ci(boot[lvl][1])
                w_lo, w_hi = ci(boot[lvl][2])
                r_lo, r_hi = ci(boot[lvl][3])
                rows.append({
                    "model": m,
                    "setting": setting,
                    "commitment": lvl,
                    "S": float(point[lvl][0]),
                    "S_ci_lo": float(s_lo),
                    "S_ci_hi": float(s_hi),
                    "RD": float(point[lvl][3]),
                    "RD_ci_lo": float(r_lo),
                    "RD_ci_hi": float(r_hi),
                    "acc_nudge_correct": float(point[lvl][1]),
                    "acc_nudge_correct_ci_lo": float(c_lo),
                    "acc_nudge_correct_ci_hi": float(c_hi),
                    "acc_nudge_wrong": float(point[lvl][2]),
                    "acc_nudge_wrong_ci_lo": float(w_lo),
                    "acc_nudge_wrong_ci_hi": float(w_hi),
                    "acc_no_nudge": acc_no,
                    "n_claims": len(df),
                    "n_unparsed_pairs": int(len(df) * len(conds) - M[..., 0].sum()),
                    "n_unparsed_no_nudge": int((~rok).sum()),
                })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "fever_scores_and_accuracy.csv", index=False)
    return out


TEMP_DATASETS = {
    "aita": {"pos": "YTA", "valid": {"yta", "nta"}},
    "lfqa": {"pos": "A", "valid": {"a", "b"}},
    "dqa": {"pos": "yes", "valid": {"yes", "no"}},
}
TEMPS = [0.25, 0.5, 0.75, 1.0]
RUNS = [1, 2, 3]


def temp_tag(t):
    return f"temp_{str(t).replace('.', '_')}"


def temp_run_path(name, temp, run):
    if run == 1:
        return OUT / "temperature_experiment" / temp_tag(temp) / f"claude_sonnet_{name}_{temp_tag(temp)}_results.csv"
    return OUT / "temperature_experiment_repeats" / f"run_{run}" / temp_tag(temp) / f"claude_sonnet_{name}_{temp_tag(temp)}_run{run}_results.csv"


def mean_s_temp(df, cfg):
    target = cfg["pos"].lower()
    vals, unparsed = [], 0
    norm = lambda x: str(x).strip().lower()
    for pc in [c for c in df.columns if c.startswith("answer_pp_plus_")]:
        nc = pc.replace("answer_pp_plus_", "answer_pp_minus_")
        if nc not in df.columns:
            continue
        a = df[pc].map(norm)
        b = df[nc].map(norm)
        unparsed += int((~a.isin(cfg["valid"])).sum() + (~b.isin(cfg["valid"])).sum())
        vals.append(np.log10(((a == target).mean() + TAU) / ((b == target).mean() + TAU)))
    return float(np.mean(vals)), unparsed


def temperature_runs():
    rows = []
    for name, cfg in TEMP_DATASETS.items():
        ref = pd.read_csv(temp_run_path(name, 0.25, 1))
        keys = set(ref["item_id"])
        base = pd.read_csv(OUT / name / "claude_sonnet_results.csv")
        matched = base[base["item_id"].isin(keys)]
        s, unp = mean_s_temp(matched, cfg)
        rows.append({"Dataset": name.upper(), "Temperature": 0.0, "Run": 1, "S": s, "n_questions": len(matched), "unparsed": unp})
        for temp in TEMPS:
            for run in RUNS:
                d = pd.read_csv(temp_run_path(name, temp, run))
                assert d["item_id"].tolist() == ref["item_id"].tolist()
                s, unp = mean_s_temp(d, cfg)
                rows.append({"Dataset": name.upper(), "Temperature": temp, "Run": run, "S": s, "n_questions": len(d), "unparsed": unp})
    return pd.DataFrame(rows)


def temp_tables():
    long = temperature_runs()
    summ = long.groupby(["Dataset", "Temperature"]).agg(
        S_mean=("S", "mean"), S_std=("S", "std"), n_runs=("S", "size"), n_questions=("n_questions", "first")
    ).reset_index()
    overall_runs = long.groupby(["Temperature", "Run"])["S"].mean().reset_index()
    overall = overall_runs.groupby("Temperature").agg(S_mean=("S", "mean"), S_std=("S", "std"), n_runs=("S", "size")).reset_index()
    long.to_csv(TABLES / "temperature_runs_long.csv", index=False)
    summ.to_csv(TABLES / "temperature_by_dataset.csv", index=False)
    overall.to_csv(TABLES / "temperature_overall.csv", index=False)
    return summ, overall


def temperature_table(summ, overall):
    cell = lambda r: f"{r.S_mean:.3f}" if r.n_runs == 1 else f"{r.S_mean:.3f} ± {r.S_std:.3f}"
    t = summ.assign(cell=summ.apply(cell, axis=1)).pivot(index="Temperature", columns="Dataset", values="cell")
    t["Overall"] = overall.set_index("Temperature").apply(cell, axis=1)
    return t[["AITA", "DQA", "LFQA", "Overall"]]


ABL_SOURCES = {
    "no_instruction": ("dqa", "{m}_results.csv"),
    "instruction_only": ("ablation_study/temp_0", "{m}_ablation_instruction_only_temp_0_results.csv"),
    "generic_cot": ("ablation_study/temp_0", "{m}_ablation_generic_cot_temp_0_results.csv"),
    "scaffold_only": ("ablation_study/temp_0", "{m}_ablation_scaffold_only_temp_0_results.csv"),
    "few_shot_1": ("ablation_study/temp_0", "{m}_ablation_few_shot_1_temp_0_results.csv"),
    "few_shot_3": ("ablation_study/temp_0", "{m}_ablation_few_shot_3_temp_0_results.csv"),
    "few_shot_5": ("ablation_study/temp_0", "{m}_ablation_few_shot_5_temp_0_results.csv"),
    "few_shot_only": ("ablation_study/temp_0", "{m}_ablation_few_shot_only_temp_0_results.csv"),
    "cf_ablated": ("ablation_study/temp_0", "{m}_ablation_cf_ablated_temp_0_results.csv"),
    "counterfactual_cot": ("dqa_mitigation_few_shot", "{m}_results.csv"),
}
DIRECT = {"instruction_only", "few_shot_only"}


def abl_matrix(df, direct):
    plus_cols = [c for c in df.columns if c.startswith("answer_pp_plus_")]
    conds, mats = [], []
    for pc in plus_cols:
        cond = pc.replace("answer_pp_plus_", "")
        mc = f"answer_pp_minus_{cond}"
        if mc not in df.columns:
            continue
        a = df[pc].map(lambda x: abl_norm(x, direct))
        b = df[mc].map(lambda x: abl_norm(x, direct))
        ok = (a.notna() & b.notna()).to_numpy()
        mats.append(np.stack([ok, ok & a.eq("yes").to_numpy(), ok & b.eq("yes").to_numpy()], axis=1).astype(float))
        conds.append(cond)
    return conds, np.stack(mats, axis=1)


def abl_group(sums, conds):
    s = s_from_counts(sums[..., 0], sums[..., 1], sums[..., 2])
    rd = (sums[..., 1] - sums[..., 2]) / sums[..., 0]
    res = {}
    for lvl in COMMIT:
        sel = [i for i, c in enumerate(conds) if c.split("_")[0] == lvl]
        res[lvl] = (np.nanmean(s[..., sel], axis=-1), np.nanmean(rd[..., sel], axis=-1))
    return res


def run_ablation(seed=1):
    reseed(seed)
    ref_df = pd.read_csv(OUT / "ablation_study/temp_0/claude_sonnet_ablation_few_shot_only_temp_0_results.csv")
    ref_q = set(ref_df["item_id"])
    assert len(ref_q) == 50, len(ref_q)
    rows = []
    for rung, (folder, pattern) in ABL_SOURCES.items():
        for m in MODEL_ORDER:
            key = ABL_KEY[m]
            path = OUT / folder / pattern.format(m=key)
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df[df["item_id"].isin(ref_q)].reset_index(drop=True)
            conds, M = abl_matrix(df, rung in DIRECT)
            point = abl_group(M.sum(axis=0), conds)
            boot = abl_group(chunked_sums(M, B_ABL), conds)
            for lvl in COMMIT:
                sel = [i for i, c in enumerate(conds) if c.split("_")[0] == lvl]
                lo, hi = ci(boot[lvl][0])
                rlo, rhi = ci(boot[lvl][1])
                n_pairs = len(df) * len(sel)
                rows.append({
                    "rung": rung,
                    "model": m,
                    "commitment": lvl,
                    "S": float(point[lvl][0]),
                    "S_ci_lo": float(lo),
                    "S_ci_hi": float(hi),
                    "RD": float(point[lvl][1]),
                    "RD_ci_lo": float(rlo),
                    "RD_ci_hi": float(rhi),
                    "n_questions": len(df),
                    "unparsed_pairs": int(n_pairs - M[:, sel, 0].sum()),
                    "total_pairs": n_pairs,
                })
    out = pd.DataFrame(rows)
    out["unparsed_frac"] = out["unparsed_pairs"] / out["total_pairs"]
    out.to_csv(TABLES / "ablation_S_with_ci.csv", index=False)
    return out


PAPER_COLORS = {
    "no_instruction": "#4878d0",
    "few_shot_only": "#ee854a",
    "cf_ablated": "#d65f5f",
    "counterfactual_cot": "#6acc64",
}
PAPER_LABELS = {
    "no_instruction": "No mitigation",
    "few_shot_only": "Examples, direct answer",
    "cf_ablated": "CoT, no counterfactual steps",
    "counterfactual_cot": "Counterfactual CoT",
}


def plot_ablation_s_main(tab, name="ablation_s_main", ci=False):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6), sharey=True)
    rungs = list(PAPER_COLORS)
    width = 0.2
    xs = np.arange(3)
    for ax, m in zip(axes.ravel(), MODEL_ORDER):
        ax.axhline(0, color="#555555", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, linestyle=(0, (2, 2)))
        ax.set_axisbelow(True)
        for i, rung in enumerate(rungs):
            d = tab[(tab.model == m) & (tab.rung == rung)].set_index("commitment").reindex(COMMIT)
            xpos = xs + (i - 1.5) * width
            ax.bar(xpos, d["S"].to_numpy(), width, color=PAPER_COLORS[rung], alpha=0.9, zorder=3)
            if ci:
                lo_raw = d["S_ci_lo"].to_numpy()
                hi_raw = d["S_ci_hi"].to_numpy()
                y = d["S"].to_numpy()
                lo_c = np.maximum(lo_raw, -0.5)
                hi_c = np.minimum(hi_raw, 2.5)
                ax.errorbar(xpos, y, yerr=[(y - lo_c).clip(min=0), (hi_c - y).clip(min=0)], fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2, zorder=4)
                for xi, h, l in zip(xpos, hi_raw, lo_raw):
                    if h > 2.5:
                        ax.scatter([xi], [2.47], marker="^", s=16, color="#333333", zorder=5, clip_on=False)
                    if l < -0.5:
                        ax.scatter([xi], [-0.47], marker="v", s=16, color="#333333", zorder=5, clip_on=False)
        ax.set_ylim(-0.5, 2.5)
        ax.set_title(DISPLAY[m], fontsize=15, fontweight="bold", pad=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(["Low", "Medium", "High"], fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("S", fontsize=14)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PAPER_COLORS[r], alpha=0.9) for r in rungs]
    fig.legend(handles, [PAPER_LABELS[r] for r in rungs], loc="upper center", ncol=4, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 1.02), handlelength=1.6, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
    save(fig, name)


def plot_fever_s_main(tab, name="fever_s_main", ci=False):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6), sharey=True)
    xs = np.arange(3)
    width = 0.34
    cols = {"plain": "#4878d0", "CoT": "#6acc64"}
    labs = {"plain": "No mitigation", "CoT": "Counterfactual CoT"}
    for ax, m in zip(axes.ravel(), MODEL_ORDER):
        ax.axhline(0, color="#555555", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, linestyle=(0, (2, 2)))
        ax.set_axisbelow(True)
        for i, setting in enumerate(("plain", "CoT")):
            d = tab[(tab.model == m) & (tab.setting == setting)].set_index("commitment").reindex(COMMIT)
            xpos = xs + (i - 0.5) * width
            ax.bar(xpos, d["S"].to_numpy(), width, color=cols[setting], alpha=0.9, zorder=3)
            if ci:
                yerr = [(d["S"] - d["S_ci_lo"]).to_numpy(), (d["S_ci_hi"] - d["S"]).to_numpy()]
                ax.errorbar(xpos, d["S"].to_numpy(), yerr=yerr, fmt="none", ecolor="#333333", elinewidth=1.1, capsize=3, zorder=4)
        ax.set_ylim(-0.25, 1.0)
        ax.set_title(DISPLAY[m], fontsize=15, fontweight="bold", pad=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(["Low", "Medium", "High"], fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("S", fontsize=14)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cols[k], alpha=0.9) for k in ("plain", "CoT")]
    fig.legend(handles, [labs[k] for k in ("plain", "CoT")], loc="upper center", ncol=2, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 1.02), handlelength=1.6, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
    save(fig, name)


def plot_temperature_main(summ):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, linestyle=(0, (2, 2)))
    ax.set_axisbelow(True)
    temps = sorted(summ.Temperature.unique())
    xs = np.arange(len(temps))
    width = 0.26
    cols = {"AITA": "#4878d0", "DQA": "#d65f5f", "LFQA": "#6acc64"}
    for i, ds in enumerate(("AITA", "DQA", "LFQA")):
        d = summ[summ.Dataset == ds].set_index("Temperature").reindex(temps)
        ax.bar(xs + (i - 1) * width, d["S_mean"].to_numpy(), width, yerr=d["S_std"].fillna(0).to_numpy(),
               color=cols[ds], alpha=0.9, capsize=2.5, error_kw={"elinewidth": 1.0, "ecolor": "#333333"}, label=ds, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{t:g}" for t in temps], fontsize=12)
    ax.set_xlabel("Sampling temperature", fontsize=12)
    ax.set_ylabel("S", fontsize=13)
    ax.set_ylim(0, 0.6)
    ax.tick_params(axis="y", labelsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    save(fig, "temperature_main")


def plot_ladder_main(tab, name="ladder_main", ci=False):
    rungs = [
        ("no_instruction", "No mitigation"),
        ("instruction_only", "Instruction only"),
        ("generic_cot", "Generic CoT (zero-shot)"),
        ("scaffold_only", "Scaffold, 0 examples"),
        ("few_shot_1", "Scaffold + 1 example"),
        ("few_shot_3", "Scaffold + 3 examples"),
        ("few_shot_5", "Scaffold + 5 examples"),
        ("few_shot_only", "10 examples, direct answer"),
        ("cf_ablated", "10 examples, CoT without counterfactual steps"),
        ("counterfactual_cot", "10 examples, counterfactual CoT"),
    ]
    d = tab[tab.model == "claude_sonnet"]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, linestyle=(0, (2, 2)))
    ax.set_axisbelow(True)
    ax.axvline(0, color="#555555", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    cols = {"low": "#4878d0", "medium": "#d65f5f", "high": "#6acc64"}
    labs = {"low": "Low", "medium": "Medium", "high": "High"}
    h = 0.26
    ys = np.arange(len(rungs))[::-1]
    for j, lv in enumerate(COMMIT):
        vals = []
        for key, _ in rungs:
            r = d[(d.rung == key) & (d.commitment == lv)]
            vals.append(float(r["S"].iloc[0]))
        yy = ys + (1 - j) * h
        ax.barh(yy, vals, h, color=cols[lv], alpha=0.9, label=labs[lv], zorder=3)
        if ci:
            v = np.array(vals)
            lo_raw = np.array([float(d[(d.rung == k) & (d.commitment == lv)]["S_ci_lo"].iloc[0]) for k, _ in rungs])
            hi_raw = np.array([float(d[(d.rung == k) & (d.commitment == lv)]["S_ci_hi"].iloc[0]) for k, _ in rungs])
            lo_c = np.maximum(lo_raw, -1.0)
            hi_c = np.minimum(hi_raw, 1.0)
            ax.errorbar(v, yy, xerr=[(v - lo_c).clip(min=0), (hi_c - v).clip(min=0)], fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2, zorder=4)
            for yi, hh, ll in zip(yy, hi_raw, lo_raw):
                if hh > 1.0:
                    ax.scatter([0.97], [yi], marker=">", s=14, color="#333333", zorder=5, clip_on=False)
                if ll < -1.0:
                    ax.scatter([-0.97], [yi], marker="<", s=14, color="#333333", zorder=5, clip_on=False)
    if ci:
        for yi, (k, _) in zip(ys, rungs):
            r = d[d.rung == k]
            n = int(r["n_questions"].iloc[0])
            um = float(r["unparsed_pairs"].sum() / r["total_pairs"].sum())
            txt = f"n={n}" + (f", {um * 100:.0f}% unparsed" if um >= 0.005 else "")
            ax.text(1.03, yi, txt, transform=ax.get_yaxis_transform(), va="center", fontsize=9.5,
                    color="#d65f5f" if um > 0.10 else "#555555")
    ax.set_yticks(ys)
    ax.set_yticklabels([n for _, n in rungs], fontsize=11)
    ax.set_xlim((-1.0, 1.0) if ci else (-0.5, 0.5))
    ax.set_xlabel("S", fontsize=13)
    ax.tick_params(axis="x", labelsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(title="Commitment", loc="upper center", ncol=3, frameon=False, fontsize=11, title_fontsize=11, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    save(fig, name)


def plot_fever_acc_main(tab, lvl="high"):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6), sharey=True)
    xs = np.arange(2)
    width = 0.24
    spec = [
        ("acc_no_nudge", None, None, "#4878d0", "No nudge"),
        ("acc_nudge_correct", "acc_nudge_correct_ci_lo", "acc_nudge_correct_ci_hi", "#6acc64", "Nudge toward the correct answer"),
        ("acc_nudge_wrong", "acc_nudge_wrong_ci_lo", "acc_nudge_wrong_ci_hi", "#d65f5f", "Nudge toward the wrong answer"),
    ]
    for ax, m in zip(axes.ravel(), MODEL_ORDER):
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, linestyle=(0, (2, 2)))
        ax.set_axisbelow(True)
        rows = [tab[(tab.model == m) & (tab.setting == st) & (tab.commitment == lvl)].iloc[0] for st in ("plain", "CoT")]
        for i, (col, lo, hi, color, _) in enumerate(spec):
            vals = np.array([r[col] for r in rows])
            xpos = xs + (i - 1) * width
            ax.bar(xpos, vals, width, color=color, alpha=0.9, zorder=3)
            if lo is not None:
                yerr = [vals - np.array([r[lo] for r in rows]), np.array([r[hi] for r in rows]) - vals]
                ax.errorbar(xpos, vals, yerr=yerr, fmt="none", ecolor="#333333", elinewidth=1.1, capsize=3, zorder=4)
        ax.set_ylim(0, 1.0)
        ax.set_title(DISPLAY[m], fontsize=15, fontweight="bold", pad=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(["No mitigation", "Counterfactual CoT"], fontsize=11)
        ax.tick_params(axis="y", labelsize=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy", fontsize=14)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.9) for _, _, _, c, _ in spec]
    fig.legend(handles, [l for *_, l in spec], loc="upper center", ncol=3, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 1.02), handlelength=1.6, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
    save(fig, f"fever_accuracy_{lvl}")


LADDER = [
    ("no_instruction", "No mitigation"),
    ("instruction_only", "Instruction only"),
    ("generic_cot", "Generic CoT (zero-shot)"),
    ("scaffold_only", "Scaffold, 0 examples"),
    ("few_shot_1", "Scaffold + 1 example"),
    ("few_shot_3", "Scaffold + 3 examples"),
    ("few_shot_5", "Scaffold + 5 examples"),
    ("few_shot_only", "10 examples, direct answer"),
    ("cf_ablated", "10 examples, CoT without counterfactual steps"),
    ("counterfactual_cot", "10 examples, counterfactual CoT"),
]
