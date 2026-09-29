#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import json
import re
from pathlib import Path

import pandas as pd

_REINVENT4_DIR = Path(__file__).resolve().parent / "reinvent4"
if str(_REINVENT4_DIR) not in sys.path:
    sys.path.insert(0, str(_REINVENT4_DIR))
from stage_for_flatten import HP_TO_EXP, SPEC_JOB_RE as HIT_JOB_RE  # noqa: E402


# ------------------------
# helpers
# ------------------------

def suffix_int(name: str) -> int:
    try:
        return int(name.split("-")[-1])
    except Exception as e:
        raise ValueError(f"Cannot parse numeric suffix from '{name}'") from e


def resolve_molecules_csv(seed_dir: str, molecules_csv_name: str | None) -> str | None:
    if molecules_csv_name:
        preferred = os.path.join(seed_dir, molecules_csv_name)
        if os.path.isfile(preferred):
            return preferred

    csvs = sorted(glob.glob(os.path.join(seed_dir, "*.csv")))
    if len(csvs) == 1:
        return csvs[0]

    return None


def normalize_name(s: str) -> str:
    return s.strip().lower()


def resolve_target_column(
    df: pd.DataFrame,
    target_folder: str,
    *,
    ds_col: str,
    target_col_overrides: dict[str, str] | None = None,
) -> str:
    """
    df columns are already lowercased
    """
    target_folder_lc = normalize_name(target_folder)
    ds_col_lc = normalize_name(ds_col)

    # 1) explicit per-target override, if provided
    if target_col_overrides and target_folder_lc in target_col_overrides:
        override_col = normalize_name(target_col_overrides[target_folder_lc])
        if override_col in df.columns:
            return override_col
        raise ValueError(
            f"Target '{target_folder}' override column '{override_col}' not found in CSV columns: {list(df.columns)}"
        )

    # 2) column named exactly like target folder
    if target_folder_lc in df.columns:
        return target_folder_lc

    # 3) fallback to ds_col (default: 'ds')
    if ds_col_lc in df.columns:
        return ds_col_lc

    raise ValueError(
        f"No target column '{target_folder_lc}' (or override) and no fallback '{ds_col_lc}' found"
    )


# ------------------------
# hit logic
# ------------------------

def hit_mask(
    df: pd.DataFrame,
    *,
    qed_col: str,
    sa_col: str,
    target_col: str,
    target_folder: str,
    ranges: dict,
) -> pd.Series:
    qed = pd.to_numeric(df[qed_col], errors="coerce")
    sa  = pd.to_numeric(df[sa_col], errors="coerce")
    tgt = pd.to_numeric(df[target_col], errors="coerce")

    return (
        (qed >= ranges["qed"]["min"]) &
        (qed <= ranges["qed"]["max"]) &
        (sa  >= ranges["sa"]["min"]) &
        (sa  <= ranges["sa"]["max"]) &
        (tgt >= ranges["target"][target_folder]) &
        (tgt <= ranges["target"]["max"])
    )


def compute_hit_rate(
    csv_path: str,
    target_folder: str,
    ranges: dict,
    *,
    qed_col: str,
    sa_col: str,
    ds_col: str,
    target_col_overrides: dict[str, str] | None,
    warn_invalid_rows: bool = True,
) -> float:
    df = pd.read_csv(csv_path)

    # normalize column names ONCE
    df.columns = [c.lower() for c in df.columns]

    qed_col_lc = normalize_name(qed_col)
    sa_col_lc = normalize_name(sa_col)

    missing = {qed_col_lc, sa_col_lc} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {csv_path}")

    target_col = resolve_target_column(
        df,
        target_folder,
        ds_col=ds_col,
        target_col_overrides=target_col_overrides,
    )

    mask = hit_mask(
        df,
        qed_col=qed_col_lc,
        sa_col=sa_col_lc,
        target_col=target_col,
        target_folder=normalize_name(target_folder),
        ranges=ranges,
    )

    valid = (
        pd.to_numeric(df[qed_col_lc], errors="coerce").notna() &
        pd.to_numeric(df[sa_col_lc], errors="coerce").notna() &
        pd.to_numeric(df[target_col], errors="coerce").notna()
    )

    denom = int(valid.sum())
    invalid = int((~valid).sum())

    if warn_invalid_rows and invalid > 0:
        print(f"[WARN] {csv_path}: {invalid} invalid rows (non-numeric/missing in {qed_col_lc}/{sa_col_lc}/{target_col})")

    hits = int((mask & valid).sum())
    return hits / denom if denom > 0 else 0.0


# ------------------------
# flattening logic
# ------------------------

def flatten_hit_rates(
    root_dir: str,
    algorithm_name: str,
    out_csv: str,
    ranges: dict,
    targets: list[str],
    molecules_csv_name: str | None,
    *,
    qed_col: str,
    sa_col: str,
    ds_col: str,
    target_col_overrides: dict[str, str] | None,
    warn_invalid_rows: bool = True,
    layout: str = "nested",
) -> pd.DataFrame:

    rows = []
    layout = (layout or "nested").strip().lower()

    if layout == "reinvent4":
        jobs_root = os.path.join(root_dir, "jobs")
        if not os.path.isdir(jobs_root):
            raise FileNotFoundError(f"No jobs/ folder under {root_dir}")
        for target in targets:
            target_path = os.path.join(jobs_root, target)
            if not os.path.isdir(target_path):
                continue
            for job_path in sorted(glob.glob(os.path.join(target_path, "*"))):
                if not os.path.isdir(job_path):
                    continue
                m = HIT_JOB_RE.match(os.path.basename(job_path))
                if not m:
                    continue
                exp_num = HP_TO_EXP.get(m.group("hp"))
                if exp_num is None:
                    continue
                seed_num = int(m.group("seed"))
                csv_path = resolve_molecules_csv(job_path, molecules_csv_name)
                if csv_path is None:
                    print(f"[SKIP] no CSV in {job_path}")
                    continue
                try:
                    hit_rate = compute_hit_rate(
                        csv_path, target, ranges,
                        qed_col=qed_col, sa_col=sa_col, ds_col=ds_col,
                        target_col_overrides=target_col_overrides,
                        warn_invalid_rows=warn_invalid_rows,
                    )
                except Exception as e:
                    print(f"[WARN] {csv_path}: {e}")
                    continue
                rows.append({
                    "algorithm": algorithm_name,
                    "exp": exp_num,
                    "target": target,
                    "seed": seed_num,
                    "hit_rate": hit_rate,
                })

    elif layout == "nested":
        exp_dirs = sorted(glob.glob(os.path.join(root_dir, "exp-*")))
        if not exp_dirs:
            raise FileNotFoundError(f"No exp-* folders under {root_dir}")

        for exp_path in exp_dirs:
            exp_num = suffix_int(os.path.basename(exp_path))

            for target in targets:
                target_path = os.path.join(exp_path, target)
                if not os.path.isdir(target_path):
                    continue

                seed_dirs = sorted(glob.glob(os.path.join(target_path, "seed-*")))
                for seed_path in seed_dirs:
                    seed_num = suffix_int(os.path.basename(seed_path))

                    csv_path = resolve_molecules_csv(seed_path, molecules_csv_name)
                    if csv_path is None:
                        print(f"[SKIP] no unique CSV in {seed_path}")
                        continue

                    try:
                        hit_rate = compute_hit_rate(
                            csv_path, target, ranges,
                            qed_col=qed_col, sa_col=sa_col, ds_col=ds_col,
                            target_col_overrides=target_col_overrides,
                            warn_invalid_rows=warn_invalid_rows,
                        )
                    except Exception as e:
                        print(f"[WARN] {csv_path}: {e}")
                        continue

                    rows.append({
                        "algorithm": algorithm_name,
                        "exp": exp_num,
                        "target": target,
                        "seed": seed_num,
                        "hit_rate": hit_rate,
                    })
    else:
        raise ValueError(f"Unknown layout '{layout}' (use 'nested' or 'reinvent4')")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["exp", "target", "seed"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


# ------------------------
# CLI
# ------------------------

def parse_target_col_overrides(s: str | None) -> dict[str, str] | None:
    """
    Expected JSON object string, e.g.:
      {"parp1": "parp1_dock", "fa7": "fa7_score", "5ht1b": "dock"}
    Keys are target folder names; values are CSV column names.
    """
    if not s:
        return None
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("must be a JSON object")
        # normalize keys but keep values (we'll lowercase for matching later)
        return {normalize_name(k): v for k, v in obj.items()}
    except Exception as e:
        raise ValueError(f"Invalid --target_col_overrides JSON: {e}") from e


def main():
    parser = argparse.ArgumentParser(description="Flatten nested experiments into hit-rate CSV")
    parser.add_argument("--root_dir", default=None, help="Root directory (default: cwd)")
    parser.add_argument("--algorithm_name", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--molecules_csv_name", default=None, help="Preferred CSV filename (optional)")

    # configurable column names
    parser.add_argument("--qed_col", default="qed", help="Column name for QED (default: qed)")
    parser.add_argument("--sa_col", default="sa", help="Column name for SA (default: sa)")
    parser.add_argument("--ds_col", default="ds", help="Fallback docking/score column name (default: ds)")

    # per-target override map (optional)
    parser.add_argument(
        "--target_col_overrides",
        default=None,
        help='JSON map target->column, e.g. \'{"parp1":"parp1_score","fa7":"score_fa7"}\'',
    )

    parser.add_argument(
        "--no_warn_invalid_rows",
        action="store_true",
        help="Disable warning prints about invalid rows in CSVs",
    )
    parser.add_argument(
        "--layout",
        choices=("nested", "reinvent4"),
        default="nested",
        help="nested: root/exp-*/<target>/seed-*/... (default). "
        "reinvent4: root/jobs/<target>/<hp_job>/molecules.csv.",
    )
    parser.add_argument("--targets", nargs="+", default=None, help="Target list (overrides default)")

    args = parser.parse_args()
    root_dir = os.getcwd() if args.root_dir is None else os.path.abspath(os.path.expanduser(args.root_dir))

    RANGES = {
        "qed": {"min": 0.5, "max": 1.0},
        "sa":  {"min": 1.0, "max": 5.0},
        "target": {
            "max": 20.0,
            "parp1": 10.0,
            "fa7": 8.5,
            "5ht1b": 8.7845,
            "jak2": 9.1,
            "braf": 10.3,
        },
    }

    TARGETS = args.targets if args.targets else ["parp1", "fa7", "5ht1b"]

    target_col_overrides = parse_target_col_overrides(args.target_col_overrides)

    df = flatten_hit_rates(
        root_dir=root_dir,
        algorithm_name=args.algorithm_name,
        out_csv=args.out_csv,
        ranges=RANGES,
        targets=TARGETS,
        molecules_csv_name=args.molecules_csv_name,
        qed_col=args.qed_col,
        sa_col=args.sa_col,
        ds_col=args.ds_col,
        target_col_overrides=target_col_overrides,
        warn_invalid_rows=(not args.no_warn_invalid_rows),
        layout=args.layout,
    )

    print(f"\nWrote {len(df)} rows → {args.out_csv}")
    if len(df) > 0:
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()



#python flatten_hit.py
  #--algorithm_name genetic_gfn
  #--out_csv out.csv
  #--target_col_overrides '{"parp1":"parp1_dock","fa7":"fa7_dock","5ht1b":"dock_5ht1b"}'
  #--qed_col QED_score
  #--sa_col SA_score
  #--ds_col DockScore

