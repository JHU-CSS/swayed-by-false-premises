import argparse
import hashlib
from pathlib import Path

import pandas as pd

KEY_COLUMNS = ["prompt", "base_prompt", "question"]
LABEL_COLUMNS = ["outcome", "ground_truth"]
MAX_ANSWER_CHARS = 16

SOURCES = [
    "aita/*_results.csv",
    "lfqa/*_results.csv",
    "dqa/*_results.csv",
    "aita_mitigation_cot/*_results.csv",
    "lfqa_mitigation_cot/*_results.csv",
    "dqa_mitigation_baseline/*_results.csv",
    "dqa_mitigation_few_shot/*_results.csv",
    "dqa_evidence/*/claude_sonnet_results.csv",
    "fever/*_results.csv",
    "ablation_study/temp_0/*_results.csv",
    "temperature_experiment/*/*_results.csv",
    "temperature_experiment_repeats/*/*/*_results.csv",
    "llm_judge/*/judge/*_results_*.csv",
]


def item_id(text):
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:16]


def slim(df):
    key = next((c for c in KEY_COLUMNS if c in df.columns), None)
    if key is None:
        raise ValueError("no key column found")
    out = pd.DataFrame({"item_id": df[key].map(item_id)})
    for c in LABEL_COLUMNS:
        if c in df.columns:
            out[c] = df[c]
    answers = [c for c in df.columns if c.startswith("answer")]
    for c in answers:
        out[c] = df[c]
    too_long = {c: int(df[c].dropna().astype(str).str.len().max()) for c in answers if df[c].dropna().astype(str).str.len().max() > MAX_ANSWER_CHARS}
    return out, too_long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    report = []
    for pattern in SOURCES:
        for src in sorted(args.raw.glob(pattern)):
            rel = src.relative_to(args.raw)
            df = pd.read_csv(src)
            slim_df, too_long = slim(df)
            dst = args.out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            slim_df.to_csv(dst, index=False)
            report.append((str(rel), len(slim_df), slim_df.shape[1], too_long))
    for rel, n, k, too_long in report:
        flag = f"  LONG ANSWERS: {too_long}" if too_long else ""
        print(f"{rel}: {n} rows, {k} cols{flag}")
    print(f"{len(report)} files written")


if __name__ == "__main__":
    main()
