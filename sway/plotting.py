import os
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MODEL_LABELS = {
    'meta':          'Llama 4',
    'mistral':       'Mistral',
    'claude_sonnet': 'Claude Sonnet',
    'claude_opus':   'Claude Opus',
    'claude_haiku':  'Claude Haiku',
    'gemma-3-4b-it': 'Gemma 3 4B IT',
}

MITIGATION_LABELS = {**MODEL_LABELS, "gemma-3-4b-it": "Gemma"}

Y_MIN = -0.5
Y_MAX = 2.5

COLORS = {"low": "#4878CF", "medium": "#D65F5F", "high": "#6AAF6A"}


def plot_avg_commitment(df, dataset_name, save_dir=None):
    score_cols = [c for c in df.columns if c not in ['dataset', 'model_name']]
    low_cols   = [c for c in score_cols if c.startswith('low_')]
    mid_cols   = [c for c in score_cols if c.startswith('medium_')]
    high_cols  = [c for c in score_cols if c.startswith('high_')]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    x      = np.arange(3)
    width  = 0.5
    labels = ['Low', 'Medium', 'High']
    models = df['model_name'].tolist()

    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharey=True)

    for ax, (_, row) in zip(axes, df.iterrows()):
        model = row['model_name']
        vals  = [
            np.mean([row[c] for c in low_cols]),
            np.mean([row[c] for c in mid_cols]),
            np.mean([row[c] for c in high_cols]),
        ]
        ax.bar(x, vals, width,
               color=[COLORS['low'], COLORS['medium'], COLORS['high']],
               alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=20)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=26, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_ylim(Y_MIN, Y_MAX)

    axes[0].set_ylabel('Average S', fontsize=20)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[c], label=c.capitalize())
        for c in ['low', 'medium', 'high']
    ]
    fig.legend(
        handles,
        ['Low', 'Medium', 'High'],
        loc='upper center',
        ncol=3,
        fontsize=11,
        bbox_to_anchor=(0.5, 1.08),
        frameon=False,
        title='Commitment Level',
        title_fontsize=16
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)

    if save_dir:
        plt.savefig(f'{save_dir}/{dataset_name.lower()}_avg_commitment.png',
                    bbox_inches='tight', dpi=150)
    plt.show()


def plot_specific_models_across_datasets_ci(df, dataset_model_map, base_path=None, filename="comparison.png"):
    clause_map = {
        "Plain Declarative": "declarative_plain",
        "Tagged Declarative": "declarative_tagged",
        "Imperative":         "imperative",
        "Interrogative":      "interrogative"
    }

    commitment_order  = ["low", "medium", "high"]
    commitment_colors = {
        "low":    "#4878CF",
        "medium": "#D65F5F",
        "high":   "#6AAF6A",
    }

    m_col = "model" if "model" in df.columns else "model_name"
    d_col = "dataset"

    n_rows = len(dataset_model_map)
    n_cols = len(clause_map)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows), squeeze=False)

    for row_idx, (dataset_name, model_name) in enumerate(dataset_model_map.items()):
        row_data = df[(df[m_col] == model_name) & (df[d_col] == dataset_name)]

        if row_data.empty:
            for col_idx in range(n_cols):
                axes[row_idx][col_idx].text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        row_series = row_data.iloc[0]

        for col_idx, (title, search_term) in enumerate(clause_map.items()):
            ax = axes[row_idx][col_idx]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            means = []
            for commit in commitment_order:
                target_col = [c for c in df.columns if commit in c and search_term in c]
                if target_col:
                    means.append(row_series[target_col[0]])
                else:
                    means.append(0)

            if all(m == 0 for m in means):
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            labels = [c.capitalize() for c in commitment_order]
            colors = [commitment_colors[c] for c in commitment_order]

            ax.bar(labels, means, color=colors, width=0.5, alpha=0.85)
            ax.axhline(0, linestyle="--", linewidth=0.8, color='black', alpha=0.5)

            ax.set_ylim(Y_MIN, Y_MAX)
            ax.tick_params(axis="both", labelsize=11)
            ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

            if row_idx == 0:
                ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    for row_idx, (dataset_name, model_name) in enumerate(dataset_model_map.items()):
        fig.text(
            0.08,
            1 - (row_idx + 0.5) / n_rows,
            f"{dataset_name}\n({model_name})\nS",
            fontsize=12,
            fontweight='bold',
            ha='right',
            va='center'
        )

    handles = [plt.Rectangle((0, 0), 1, 1, color=commitment_colors[c]) for c in commitment_order]
    fig.legend(
        handles,
        [c.capitalize() for c in commitment_order],
        title="Commitment Level",
        loc="upper center",
        ncol=3,
        fontsize=12,
        title_fontsize=13,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False
    )

    plt.subplots_adjust(
        top=0.82,
        bottom=0.1,
        left=0.18,
        right=0.95,
        wspace=0.3,
        hspace=0.6
    )

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_mitigation_comparison(baseline_df, mitigation_df, cot_df,
                              dataset_name, save_dir):

    score_cols = [c for c in baseline_df.columns if c not in ['dataset', 'model_name']]
    low_cols   = [c for c in score_cols if c.startswith('low_')]
    mid_cols   = [c for c in score_cols if c.startswith('medium_')]
    high_cols  = [c for c in score_cols if c.startswith('high_')]
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    x      = np.arange(3)
    width  = 0.25
    labels = ['Low', 'Medium', 'High']

    colors_three = {
        'Baseline':             '#4878CF',
        'Baseline Mitigation':  '#D65F5F',
        'CoT Mitigation':       '#6AAF6A',
    }

    models = baseline_df['model_name'].tolist()

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows),
                             sharey=True)
    axes = axes.flatten()

    for i, model in enumerate(models):
        ax = axes[i]

        b_row = baseline_df[baseline_df['model_name'] == model]
        m_row = mitigation_df[mitigation_df['model_name'] == model]
        c_row = cot_df[cot_df['model_name'] == model]

        if b_row.empty or m_row.empty or c_row.empty:
            print(f'Skipping {model} — missing data')
            continue

        b_row = b_row.iloc[0]
        m_row = m_row.iloc[0]
        c_row = c_row.iloc[0]

        b_vals = [np.mean([b_row[c] for c in low_cols]),
                  np.mean([b_row[c] for c in mid_cols]),
                  np.mean([b_row[c] for c in high_cols])]

        m_vals = [np.mean([m_row[c] for c in low_cols]),
                  np.mean([m_row[c] for c in mid_cols]),
                  np.mean([m_row[c] for c in high_cols])]

        c_vals = [np.mean([c_row[c] for c in low_cols]),
                  np.mean([c_row[c] for c in mid_cols]),
                  np.mean([c_row[c] for c in high_cols])]

        ax.bar(x - width, b_vals, width,
               color=colors_three['Baseline'], alpha=0.85)

        ax.bar(x, m_vals, width,
               color=colors_three['Baseline Mitigation'], alpha=0.85)

        ax.bar(x + width, c_vals, width,
               color=colors_three['CoT Mitigation'], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=20)
        ax.tick_params(axis='y', labelsize=20)

        ax.set_title(MITIGATION_LABELS.get(model, model),
                     fontsize=20, fontweight='bold')

        ax.axhline(0, color='black', linewidth=0.8,
                   linestyle='--', alpha=0.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.grid(axis='y', linestyle='--',
                linewidth=0.5, alpha=0.5)

        ax.set_ylim(Y_MIN, Y_MAX)

    for j in range(len(models), len(axes)):
        axes[j].set_visible(False)

    axes[0].set_ylabel('S', fontsize=20)
    axes[ncols].set_ylabel('S', fontsize=20)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c)
        for c in colors_three.values()
    ]

    fig.legend(
        handles,
        list(colors_three.keys()),
        loc='upper center',
        ncol=3,
        fontsize=16,
        bbox_to_anchor=(0.5, 1.03),
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.35)

    plt.savefig(save_dir / "mitigation_comparison.png",
                bbox_inches='tight', dpi=300)
    plt.show()


def plot_ci(results_df, dataset_name, save_dir=None, figsize=(14, 3.5)):
    order  = ["low", "medium", "high"]
    colors = {"low": "#4878CF", "medium": "#D65F5F", "high": "#6AAF6A"}

    clause_groups = {
        "Plain Declarative":  lambda c: "declarative_plain" in c,
        "Tagged Declarative": lambda c: "declarative_tagged" in c,
        "Imperative":         lambda c: "imperative" in c,
        "Interrogative":      lambda c: "interrogative" in c,
    }

    m_col = "model" if "model" in results_df.columns else "model_name"
    ds_df = results_df[results_df["dataset"] == dataset_name]

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for model in sorted(ds_df[m_col].unique()):
        model_df = ds_df[ds_df[m_col] == model].copy()

        fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)

        for ax, (clause_label, clause_filter) in zip(axes, clause_groups.items()):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            clause_df = model_df[model_df["clause_type"].apply(clause_filter)].copy()
            clause_df["commitment"] = pd.Categorical(
                clause_df["commitment"], categories=order, ordered=True
            )
            clause_df = clause_df.sort_values("commitment")

            if clause_df.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11)
                continue

            for i, (_, row) in enumerate(clause_df.iterrows()):
                color = colors.get(row["commitment"], "gray")
                ax.errorbar(
                    i, row["S"],
                    yerr=[[row["S"] - row["ci_lower"]], [row["ci_upper"] - row["S"]]],
                    fmt="o", color=color, capsize=0,
                    linewidth=2, markersize=10
                )

            ax.set_xticks(range(len(clause_df)))
            ax.set_xticklabels(
                [r["commitment"].capitalize() for _, r in clause_df.iterrows()],
                fontsize=11
            )
            ax.tick_params(axis='y', labelsize=11)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_title(clause_label, fontsize=12, fontweight='bold')
            ax.set_xlabel("Commitment Level", fontsize=11)
            ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_ylim(-0.5, 3.0)

        axes[0].set_ylabel("S", fontsize=12)

        handles = [
            mpatches.Patch(color=c, label=l.capitalize())
            for l, c in colors.items()
        ]
        handles.append(plt.Line2D(
            [0], [0], color="black", linestyle="--",
            linewidth=0.8, label="S=0 (no sycophancy)"
        ))

        fig.legend(
            handles=handles, title="Commitment",
            loc="upper center", ncol=4,
            fontsize=11, title_fontsize=11,
            bbox_to_anchor=(0.5, 1.08), frameon=False
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        if save_dir:
            safe_name = str(model).lower().replace(" ", "_")
            plt.savefig(save_dir / f"{safe_name}_ci.png",
                        bbox_inches='tight', dpi=150)
        plt.show()
        plt.close(fig)


def plot_avg_commitment_ci(results_df, dataset_name, save_dir=None, figsize=(4, 3.5)):
    order  = ["low", "medium", "high"]
    colors = {"low": "#4878CF", "medium": "#D65F5F", "high": "#6AAF6A"}

    ds_df = results_df[results_df["dataset"] == dataset_name]
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    agg = ds_df.groupby("commitment").agg(
        mean_S=("S", "mean"),
        ci_lower=("ci_lower", "mean"),
        ci_upper=("ci_upper", "mean"),
    ).reindex(order).reset_index()

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, row in agg.iterrows():
        if pd.isna(row["mean_S"]):
            continue
        color = colors.get(row["commitment"], "gray")
        ax.errorbar(
            i, row["mean_S"],
            yerr=[[row["mean_S"] - row["ci_lower"]],
                  [row["ci_upper"] - row["mean_S"]]],
            fmt="o", color=color, capsize=0,
            linewidth=2, markersize=10
        )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([c.capitalize() for c in order], fontsize=11)
    ax.set_ylabel("Average S", fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylim(-0.5, 3.0)
    ax.set_xlabel("Commitment Level", fontsize=11)

    plt.tight_layout()

    if save_dir:
        safe_name = dataset_name.lower()
        plt.savefig(save_dir / f"{safe_name}_avg_commitment_ci.png",
                    bbox_inches='tight', dpi=150)

    plt.show()
    plt.close(fig)


def plot_aita_cot_comparison(baseline_df, cot_df, dataset_name, save_dir=None):
    score_cols = [c for c in baseline_df.columns if c not in ['dataset', 'model_name']]
    low_cols   = [c for c in score_cols if c.startswith('low_')]
    mid_cols   = [c for c in score_cols if c.startswith('medium_')]
    high_cols  = [c for c in score_cols if c.startswith('high_')]
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    models = baseline_df['model_name'].tolist()
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3), sharey=True)

    if n_models == 1:
        axes = [axes]

    x = np.arange(3)
    width = 0.3
    labels = ['Low', 'Medium', 'High']

    colors = {
        'Baseline': '#4878CF',
        'CoT':      '#6AAF6A',
    }

    for ax, model in zip(axes, models):
        b_row = baseline_df[baseline_df['model_name'] == model]
        c_row = cot_df[cot_df['model_name'] == model]

        if b_row.empty or c_row.empty:
            ax.set_visible(False)
            continue

        b_row = b_row.iloc[0]
        c_row = c_row.iloc[0]

        b_vals = [np.mean([b_row[c] for c in low_cols]),
                  np.mean([b_row[c] for c in mid_cols]),
                  np.mean([b_row[c] for c in high_cols])]
        c_vals = [np.mean([c_row[c] for c in low_cols]),
                  np.mean([c_row[c] for c in mid_cols]),
                  np.mean([c_row[c] for c in high_cols])]

        ax.bar(x - width/2, b_vals, width, label='Baseline',       color=colors['Baseline'], alpha=0.85)
        ax.bar(x + width/2, c_vals, width, label='CoT Mitigation',  color=colors['CoT'],      alpha=0.85)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_title(MITIGATION_LABELS.get(model, model), fontsize=13, fontweight='bold')
        ax.set_xlabel('Commitment Level', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_ylim(-0.5, 1.8)

    axes[0].set_ylabel('Avg Sycophancy Score (S)', fontsize=12)

    handles = [
        mpatches.Patch(color=colors['Baseline'], label='Baseline'),
        mpatches.Patch(color=colors['CoT'],      label='CoT Mitigation'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               fontsize=12, bbox_to_anchor=(0.5, 1.08), frameon=False)

    plt.tight_layout()

    if save_dir:
        safe_name = dataset_name.lower().replace(' ', '_')
        plt.savefig(save_dir / f"{safe_name}_cot_comparison.png",
                    bbox_inches='tight', dpi=150)

    plt.show()
    plt.close(fig)


EVIDENCE_COLORS = {
    "Normal CoT":       "#4878CF",
    "Evidence for":     "#D65F5F",
    "Evidence against": "#6AAF6A",
}

CLAUSE_GROUPS = {
    "Plain Declarative":  "declarative_plain",
    "Tagged Declarative": "declarative_tagged",
    "Imperative":         "imperative",
    "Interrogative":      "interrogative",
}

def plot_evidence_commitment(cot_df, for_df, against_df, save_dir=None, figsize=(9, 3), y_min=-0.5, y_max=0.5):
    score_cols = [c for c in cot_df.columns if c not in ['dataset', 'model_name']]
    low_cols   = [c for c in score_cols if c.startswith('low_')]
    mid_cols   = [c for c in score_cols if c.startswith('medium_')]
    high_cols  = [c for c in score_cols if c.startswith('high_')]

    conditions = {
        "Normal CoT":       cot_df,
        "Evidence for":     for_df,
        "Evidence against": against_df,
    }

    x      = np.arange(3)
    width  = 0.25
    offset = [-width, 0, width]
    labels = ['Low', 'Medium', 'High']

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (label, df) in enumerate(conditions.items()):
        row  = df.iloc[0]
        vals = [
            np.mean([row[c] for c in low_cols]),
            np.mean([row[c] for c in mid_cols]),
            np.mean([row[c] for c in high_cols]),
        ]
        ax.bar(x + offset[i], vals, width, label=label, color=EVIDENCE_COLORS[label], alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_xlabel("Commitment Level", fontsize=12)
    ax.set_ylabel("Avg Sycophancy Score (S)", fontsize=12)
    ax.set_title("DQA — Evidence Comparison (Claude Sonnet)", fontsize=13, fontweight='bold')
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "evidence_commitment.png",
                    dpi=300, bbox_inches='tight')
    plt.show()


def plot_evidence_clause(cot_df, for_df, against_df, save_dir=None, figsize=(9, 3), y_min=-0.5, y_max=0.5):
    conditions = {
        "Normal CoT":       cot_df,
        "Evidence for":     for_df,
        "Evidence against": against_df,
    }

    clause_labels = list(CLAUSE_GROUPS.keys())
    x      = np.arange(len(clause_labels))
    width  = 0.25
    offset = [-width, 0, width]

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (label, df) in enumerate(conditions.items()):
        row  = df.iloc[0]
        vals = []
        for search_term in CLAUSE_GROUPS.values():
            cols = [c for c in df.columns if search_term in c and c not in ['dataset', 'model_name']]
            vals.append(np.mean([row[c] for c in cols]))
        ax.bar(x + offset[i], vals, width, label=label, color=EVIDENCE_COLORS[label], alpha=0.85)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(clause_labels, rotation=15, ha='right', fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_xlabel("Clause Type", fontsize=12)
    ax.set_ylabel("Avg Sycophancy Score (S)", fontsize=12)
    ax.set_title("DQA — Evidence Comparison by Clause Type (Claude Sonnet)", fontsize=13, fontweight='bold')
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "evidence_clause.png",
                    dpi=300, bbox_inches='tight')
    plt.close()


def plot_response_rates_by_clause_commitment(for_path, against_path, pos_label="yes", save_dir=None, figsize=(20, 12)):
    conditions = {
        "Evidence for":     pd.read_csv(for_path),
        "Evidence against": pd.read_csv(against_path),
    }
    condition_colors = {
        "Evidence for":     "#D65F5F",
        "Evidence against": "#6AAF6A",
    }
    commitment_levels = ["low", "medium", "high"]
    n_clauses         = len(CLAUSE_GROUPS)

    fig, axes = plt.subplots(2, n_clauses, figsize=figsize, sharey=True)
    fig.suptitle("DQA — Response Rates by Clause Type & Commitment (Claude Sonnet)", fontsize=20, fontweight='bold', y=1.02)

    x      = np.arange(len(commitment_levels))
    width  = 0.3
    offset = [-width/2, width/2]

    for pp_idx, (pp_type, col_filter) in enumerate([("PP+", "_pp_plus_"), ("PP-", "_pp_minus_")]):
        for cl_idx, (clause_label, search_term) in enumerate(CLAUSE_GROUPS.items()):
            ax = axes[pp_idx][cl_idx]
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for i, (label, df) in enumerate(conditions.items()):
                vals = []
                for commit in commitment_levels:
                    cols = [c for c in df.columns if c.startswith("answer_") and col_filter in c and search_term in c and commit in c]
                    p    = df[cols].apply(lambda c: (c.dropna().astype(str).str.lower() == pos_label).mean()).mean()
                    vals.append(p)

                bars = ax.bar(x + offset[i], vals, width, label=label, color=condition_colors[label], alpha=0.85)

                for bar in bars:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{bar.get_height()*100:.0f}%",
                        ha='center', va='bottom', fontsize=15, rotation=0
                    )

            ax.set_xticks(x)
            ax.set_xticklabels([c.capitalize() for c in commitment_levels], fontsize=20)
            ax.tick_params(axis='y', labelsize=12)
            ax.set_ylim(0, 1.2)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
            ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axhline(0.5, color='black', linewidth=0.8, linestyle=':', alpha=0.5)

            if pp_idx == 0:
                ax.set_title(clause_label, fontsize=20, fontweight='bold', pad=20)
            if cl_idx == 0:
                ax.set_ylabel(f"{pp_type} — % '{pos_label}'", fontsize=20)
            if pp_idx == 0 and cl_idx == 0:
                ax.legend(fontsize=12)

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "evidence_response_rates_clause_commitment.png",
                    dpi=300, bbox_inches='tight')
    plt.close()


def plot_response_rates_by_clause(cot_path, for_path, against_path, pos_label="yes", save_dir=None, figsize=(12, 4)):
    conditions = {
        "Normal CoT":       pd.read_csv(cot_path),
        "Evidence for":     pd.read_csv(for_path),
        "Evidence against": pd.read_csv(against_path),
    }

    condition_colors = {
        "Normal CoT":       "#4878CF",
        "Evidence for":     "#D65F5F",
        "Evidence against": "#6AAF6A",
    }

    clause_labels = list(CLAUSE_GROUPS.keys())
    n_clauses     = len(clause_labels)
    n_conditions  = len(conditions)

    x      = np.arange(n_clauses)
    width  = 0.12
    offsets_plus  = [-3*width, -width, width]
    offsets_minus = [-2*width, 0, 2*width]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle("DQA — Response Rates by Clause Type (Claude Sonnet)", fontsize=13, fontweight='bold')

    for ax, (pp_type, offsets, col_filter) in zip(axes, [
        ("PP+", offsets_plus,  "_pp_plus_"),
        ("PP-", offsets_minus, "_pp_minus_"),
    ]):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for i, (label, df) in enumerate(conditions.items()):
            answer_cols = [c for c in df.columns if c.startswith("answer_") and col_filter in c]
            vals = []
            for search_term in CLAUSE_GROUPS.values():
                cols = [c for c in answer_cols if search_term in c]
                p    = df[cols].apply(lambda c: (c.dropna().astype(str).str.lower() == pos_label).mean()).mean()
                vals.append(p)

            bars = ax.bar(x + offsets[i], vals, width, label=label, color=condition_colors[label], alpha=0.85)

            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height()*100:.0f}%",
                    ha='center', va='bottom', fontsize=8, rotation=90
                )

        ax.set_xticks(x)
        ax.set_xticklabels(clause_labels, rotation=15, ha='right', fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_ylabel(f"% '{pos_label}' responses", fontsize=12)
        ax.set_title(pp_type, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=9)

    plt.tight_layout()

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "evidence_response_rates_clause.png",
                    dpi=300, bbox_inches='tight')
    plt.close()
