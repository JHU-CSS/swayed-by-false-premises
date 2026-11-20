import numpy as np
import pandas as pd


def compare(got, expected, on, cols, atol=1e-9, label=""):
    j = got.merge(expected, on=on, how="outer", suffixes=("", "_expected"), indicator=True)
    missing = int((j["_merge"] != "both").sum())
    both = j[j["_merge"] == "both"]
    diffs = {}
    for c in cols:
        a = pd.to_numeric(both[c], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(both[f"{c}_expected"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(a) & np.isfinite(b)
        same_nonfinite = (~np.isfinite(a)) & (~np.isfinite(b))
        if (~finite & ~same_nonfinite).any():
            diffs[c] = np.inf
        else:
            diffs[c] = float(np.abs(a[finite] - b[finite]).max()) if finite.any() else 0.0
    worst = max(diffs.values()) if diffs else 0.0
    status = "OK" if missing == 0 and worst <= atol else "MISMATCH"
    print(f"[{status}] {label}: {len(both)} rows matched, {missing} unmatched, max abs diff {worst:.2e} (tolerance {atol:g})")
    assert status == "OK", diffs
    return j


def compare_wide(got, expected, key, atol=1e-9, label=""):
    cols = [c for c in expected.columns if c != key and c in got.columns and pd.api.types.is_numeric_dtype(expected[c])]
    return compare(got[[key] + cols], expected[[key] + cols], [key], cols, atol, label)
