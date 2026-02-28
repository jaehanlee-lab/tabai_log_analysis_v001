"""
merge_and_analysis.py

Purpose
-------
Offline merge + analysis utility for training logs split across multiple .wandb files.

This script:
1) Uses two functions from wandb2data.py in the same project:
   - parse_wandb_logs_to_df(path_log, dedup_by_step=True) -> pd.DataFrame
   - save_df_to_csv(df, path_csv) -> None

2) Merges multiple segments of logs created by interrupted training runs.
   Each .wandb file has step starting from 0, so we offset steps by parse_iters.

3) Truncates each segment to the requested range before merging.

4) Computes moving averages for ce and accuracy with configurable window sizes
   (default windows: 100 and 1000), then saves the result to CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Assumes wandb2data.py is in the same project directory / import path.
from wandb2data import parse_wandb_logs_to_df, save_df_to_csv


# =============================================================================
# Merge (segmentation) logic
# =============================================================================

def _validate_inputs(path_logs: Sequence[str], parse_iters: Sequence[int]) -> None:
    if not isinstance(path_logs, (list, tuple)) or not isinstance(parse_iters, (list, tuple)):
        raise TypeError("path_logs와 parse_iters는 list/tuple이어야 합니다.")
    if len(path_logs) == 0:
        raise ValueError("path_logs가 비어있습니다.")
    if len(path_logs) != len(parse_iters):
        raise ValueError(f"길이 불일치: len(path_logs)={len(path_logs)} vs len(parse_iters)={len(parse_iters)}")
    if any((not isinstance(x, int)) or (x < 0) for x in parse_iters):
        raise ValueError("parse_iters는 0 이상의 int들로만 구성되어야 합니다.")
    if list(parse_iters) != sorted(parse_iters):
        raise ValueError("parse_iters는 오름차순으로 정렬되어야 합니다.")


def merge_wandb_segments(
    path_logs: Sequence[str],
    parse_iters: Sequence[int],
    dedup_by_step: bool = True,
) -> pd.DataFrame:
    """
    Merge multiple .wandb logs into a single DataFrame according to parse_iters.

    Definitions
    ----------
    - path_logs[i] corresponds to global iteration range:
        [parse_iters[i], parse_iters[i+1]) for i < n-1
        [parse_iters[i], +inf)           for i == n-1

    - Each individual .wandb file has local step starting from 0.
      We convert local step -> global step by:
        global_step = local_step + parse_iters[i]

    - If a file contains more steps than the requested segment length,
      steps beyond the segment end are removed before merging.

    Returns
    -------
    pd.DataFrame with columns:
      step, total_step, s_per_it, accuracy, ce
    where step is global step (int).
    """
    _validate_inputs(path_logs, parse_iters)

    merged_parts: List[pd.DataFrame] = []
    n = len(path_logs)

    for i, (p, start_iter) in enumerate(zip(path_logs, parse_iters)):
        end_iter: Optional[int] = parse_iters[i + 1] if i < n - 1 else None

        df = parse_wandb_logs_to_df(p, dedup_by_step=dedup_by_step).copy()

        if df.empty:
            continue

        # local step -> int, then shift to global
        df["step"] = df["step"].astype(int) + int(start_iter)

        # truncate to requested segment range
        if end_iter is not None:
            df = df[df["step"] < int(end_iter)]

        merged_parts.append(df)

    if not merged_parts:
        # Return empty, but with stable schema
        return pd.DataFrame(columns=["step", "total_step", "s_per_it", "accuracy", "ce"])

    out = pd.concat(merged_parts, ignore_index=True)

    # Ensure unique ordering; if overlaps exist, keep the first occurrence by step
    out = out.sort_values("step", kind="mergesort")
    out = out.drop_duplicates(subset=["step"], keep="first").reset_index(drop=True)

    return out


# =============================================================================
# Analysis logic (moving averages)
# =============================================================================

def add_moving_averages(
    df: pd.DataFrame,
    windows: Sequence[int] = (100, 1000),
    cols: Sequence[str] = ("ce", "accuracy"),
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Add moving average columns for given metrics.

    For each col in cols and each window in windows, adds:
      f"{col}_ma{window}"

    Notes
    -----
    - Uses simple rolling mean.
    - min_periods=1 => early part will be averaged over available points.

    Returns a NEW DataFrame (does not mutate input).
    """
    if df is None:
        raise ValueError("df is None")
    if not windows:
        raise ValueError("windows가 비어있습니다.")
    if any((not isinstance(w, int)) or (w <= 0) for w in windows):
        raise ValueError("windows는 양의 int여야 합니다.")

    out = df.copy()

    # Guarantee step ordering for rolling computations
    if "step" in out.columns:
        out = out.sort_values("step", kind="mergesort").reset_index(drop=True)

    for c in cols:
        if c not in out.columns:
            # If a requested column is missing, create it as NaN to keep schema stable
            out[c] = pd.NA

        for w in windows:
            out[f"{c}_ma{w}"] = out[c].rolling(window=w, min_periods=min_periods).mean()

    return out

# ===== [ADD] function section =====
def plot_columns_to_png(
    df: pd.DataFrame,
    path_png: str,
    cols: Sequence[str],
    x_col: str = "step",
    title: Optional[str] = None,
    y_lim = [0.5, 1.0],
) -> None:
    """
    Plot multiple columns (lines) against x_col and save as PNG.

    Parameters
    ----------
    df      : input DataFrame
    path_png: output png path
    cols    : list/tuple of column names to plot
    x_col   : x-axis column (default: 'step')
    title   : optional plot title
    """
    if df is None or df.empty:
        raise ValueError("df가 비어있습니다.")
    if not cols:
        raise ValueError("cols가 비어있습니다.")
    if x_col not in df.columns:
        raise ValueError(f"x_col='{x_col}' 컬럼이 df에 없습니다.")

    # keep only existing columns; fail fast if none exist
    cols_exist = [c for c in cols if c in df.columns]
    if not cols_exist:
        raise ValueError(f"요청한 cols가 df에 없습니다. cols={list(cols)}")

    out_path = Path(path_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = df.sort_values(x_col, kind="mergesort")

    plt.figure()
    for c in cols_exist:
        plt.plot(d[x_col], d[c], label=c)
    plt.xlabel(x_col)
    plt.ylabel("value")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# =============================================================================
# CLI entrypoint / test
# =============================================================================

if __name__ == "__main__":
    # -------------------------
    # Test configuration
    # -------------------------
    # Example:
    #   - 3 wandb files from interrupted runs
    #   - parse_iters specify the global step start for each file
    #
    # Meaning:
    #   file0: global steps [0, 10000)
    #   file1: global steps [10000, 25000)
    #   file2: global steps [25000, +inf)
    path_logs = [
        r"/path/to/run_part0.wandb",
        r"/path/to/run_part1.wandb",
        r"/path/to/run_part2.wandb",
    ]
    parse_iters = [0, 10000, 25000]
    # Output CSV
    path_csv = r"/path/to/merged_with_ma.csv"
	
    """
	# Examples
	
    path_logs = [
        r"mlp_scm_v008/wandb/offline-run-20260227_043911-vastai000/run-vastai000.wandb",
        r"mlp_scm_v008/wandb/offline-run-20260228_032324-vastai000/run-vastai000.wandb",
    ]
    parse_iters = [0, 20000]
    # Output CSV
    path_csv = "result3.csv"
	"""

    # Moving average windows (defaults: 100, 1000)
    ma_windows = (100, 1000, 5000)

    # -------------------------
    # Run merge + analysis + save
    # -------------------------
    merged_df = merge_wandb_segments(path_logs=path_logs, parse_iters=parse_iters, dedup_by_step=True)
    analyzed_df = add_moving_averages(merged_df, windows=ma_windows, cols=("ce", "accuracy"), min_periods=1)
    save_df_to_csv(analyzed_df, path_csv)

    print(f"Saved: {path_csv}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Analyzed rows: {len(analyzed_df)}")
    print(analyzed_df.head(10).to_string(index=False))

    # ===== [ADD] __main__ section (after analyzed_df 생성/저장 라인 근처) =====
    # 1) accuracy moving averages plot
    acc_cols = [f"accuracy_ma{w}" for w in ma_windows]
    plot_columns_to_png(
	    analyzed_df,
	    path_png=str(Path(path_csv).with_suffix("")) + "_accuracy_ma.png",
	    cols=acc_cols,
	    title="Accuracy Moving Averages",
        y_lim=[0.62, 0.72],
    )

    # 2) ce moving averages plot
    ce_cols = [f"ce_ma{w}" for w in ma_windows]
    plot_columns_to_png(
	    analyzed_df,
	    path_png=str(Path(path_csv).with_suffix("")) + "_ce_ma.png",
	    cols=ce_cols,
	    title="CE Moving Averages",
        y_lim=[0.70, 0.95],
    )