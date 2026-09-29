#!/usr/bin/env python3
import os
import sys
import glob
import argparse
from pathlib import Path
import pandas as pd

_REINVENT4_DIR = Path(__file__).resolve().parent / "reinvent4"
if str(_REINVENT4_DIR) not in sys.path:
    sys.path.insert(0, str(_REINVENT4_DIR))
from stage_for_flatten import HP_TO_EXP, LEAD_JOB_RE  # noqa: E402


# ------------------------
# default hit configuration (UPDATED)
# ------------------------
DEFAULT_TARGET_BY_SEEDMOL = {
    "parp1": {0: 7.3, 1: 7.8, 2: 8.2},
    "fa7":   {0: 6.4, 1: 6.7, 2: 8.5},
    "5ht1b": {0: 4.5, 1: 7.6, 2: 9.8},
    "jak2": {0: 7.1, 1: 8.2, 2: 9.3},
    "braf": {0: 6.5, 1: 7.6, 2: 8.7},
}

RANGES = {
    "qed": {"min": 0.5, "max": 1.0},
    "sa": {"min": 1.0, "max": 5.0},
    "similarity": {"min": 0.4},
    # only keep max here; per-target/per-seedmol mins come from DEFAULT_TARGET_BY_SEEDMOL
    "target": {"max": 20.0},
}


# ------------------------
# helpers
# ------------------------
def suffix_int(name: str) -> int:
    """Extract integer suffix from exp-3 / seed-12 / start_mol-idx-7."""
    try:
        return int(name.split("-")[-1])
    except Exception as e:
        raise ValueError(f"Cannot parse numeric suffix from '{name}'") from e


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def sim_thr_to_folder(sim_thr: float) -> str:
    """Map 0.4 -> sim04, 0.6 -> sim06 (REINVENT4 grid layout)."""
    code = int(round(float(sim_thr) * 10))
    return f"sim{code:02d}"


def resolve_seed_csv(seed_dir: str, molecules_csv_name: str) -> str | None:
    """
    New rule: molecules_csv_name is required (user will pass it).
    Return that file if it exists, else None.
    """
    preferred = os.path.join(seed_dir, molecules_csv_name)
    return preferred if os.path.isfile(preferred) else None


def resolve_target_column(
    df: pd.DataFrame,
    target_folder: str,
    docking_col: str | None,
) -> str:
    """
    Decide which docking column to use.
    Priority:
      1) explicit --docking_col if provided
      2) column matching target folder name (lowercased)
      3) 'ds' if present (legacy convenience)
    """
    cols = set(df.columns)

    if docking_col:
        c = docking_col.strip().lower()
        if c in cols:
            return c
        raise ValueError(f"Docking column '{docking_col}' not found. Available: {sorted(cols)}")

    t = target_folder.lower()
    if t in cols:
        return t
    if "ds" in cols:
        return "ds"
    raise ValueError(f"No docking column provided and neither '{t}' nor 'ds' found. Available: {sorted(cols)}")


# ------------------------
# scoring logic (ONLY hit criteria part UPDATED)
# ------------------------
def compute_top_score(
    csv_path: str,
    target_folder: str,
    seedmolindex: int,
    ranges: dict,
    colmap: dict,
) -> float:
    """
    Returns lowest docking score among hit molecules.
    If no hits -> returns target default threshold.
    """
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    # Required columns are configurable (qed/sa/similarity)
    qed_col = colmap["qed"].lower()
    sa_col = colmap["sa"].lower()
    sim_col = colmap["similarity"].lower()

    missing = {qed_col, sa_col, sim_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} (after lowercasing)")

    docking_col = resolve_target_column(df, target_folder, colmap.get("docking"))

    qed = pd.to_numeric(df[qed_col], errors="coerce")
    sa = pd.to_numeric(df[sa_col], errors="coerce")
    sim = pd.to_numeric(df[sim_col], errors="coerce")
    tgt = pd.to_numeric(df[docking_col], errors="coerce")

    valid = qed.notna() & sa.notna() & sim.notna() & tgt.notna()

    # NOTE: seedmolindex must come from the folder `start_mol_idx-*` (NOT `seed-*`).
    # It is passed in by the caller (derived from the directory traversal).
    tgt_min = float(DEFAULT_TARGET_BY_SEEDMOL[target_folder][seedmolindex])
    tgt_max = float(ranges["target"]["max"])

    if valid.sum() == 0:
        print(f"[WARN] No valid rows found in {csv_path}")
        return tgt_min  # fallback uses the per-seedmol threshold

    hit = (
        valid
        & (qed >= ranges["qed"]["min"]) & (qed <= ranges["qed"]["max"])
        & (sa >= ranges["sa"]["min"]) & (sa <= ranges["sa"]["max"])
        & (sim >= ranges["similarity"]["min"])
        & (tgt <= tgt_max)
        & (tgt >= tgt_min)
    )

    if hit.sum() == 0:
        return tgt_min  # fallback uses the per-seedmol threshold

    return float(tgt[hit].max())


def _walk_nested_flat(
    rows: list,
    exp_num: int,
    exp_path: str,
    targets: list[str],
    sim_thr_dir: str,
    start_mol_folder_base: int,
    molecules_csv_name: str,
    algorithm_name: str,
    ranges: dict,
    colmap: dict,
) -> None:
    for target in targets:
        target_path = os.path.join(exp_path, target)
        if not os.path.isdir(target_path):
            continue

        sim_path = os.path.join(target_path, sim_thr_dir)
        if not os.path.isdir(sim_path):
            continue

        start_mol_dirs = sorted(
            glob.glob(os.path.join(sim_path, "start_mol_idx-*"))
        )
        if not start_mol_dirs:
            continue

        for sm_path in start_mol_dirs:
            sm_idx = suffix_int(os.path.basename(sm_path))
            seedmolindex = sm_idx - int(start_mol_folder_base)

            seed_dirs = sorted(glob.glob(os.path.join(sm_path, "seed-*")))
            for seed_path in seed_dirs:
                seed_num = suffix_int(os.path.basename(seed_path))

                csv_path = resolve_seed_csv(seed_path, molecules_csv_name)
                if csv_path is None:
                    print(f"[SKIP] missing '{molecules_csv_name}' in {seed_path}")
                    continue

                try:
                    top_score = compute_top_score(
                        csv_path,
                        target,
                        seedmolindex,
                        ranges,
                        colmap,
                    )
                except Exception as e:
                    print(f"[WARN] {csv_path}: {e}")
                    continue

                rows.append(
                    {
                        "algorithm": algorithm_name,
                        "exp": exp_num,
                        "target": target,
                        "seedmolindex": seedmolindex,
                        "seed": seed_num,
                        "top_score": top_score,
                    }
                )


# ------------------------
# flatten logic (NEW hierarchy)
# ------------------------
def flatten_top_scores(
    root_dir: str,
    algorithm_name: str,
    out_csv: str,
    ranges: dict,
    targets: list[str],
    molecules_csv_name: str,
    colmap: dict,
    *,
    layout: str = "nested",
    flat_exp_index: int = 0,
    sim_thr_dir: str = "sim_thr-0.6",
    start_mol_folder_base: int = 1,
    sim_thr: float | None = None,
) -> pd.DataFrame:
    """
    Walks (nested):
      exp-*/<target>/sim_thr-0.6/start_mol_idx-*/seed-*/<molecules_csv_name>

    Walks (flat):
      <root>/<target>/sim_thr-0.6/start_mol_idx-*/seed-*/<molecules_csv_name>

    Walks (reinvent4):
      jobs/<target>/sim04|sim06/mol*/<hp_job>/molecules.csv

    Output:
      algorithm, exp, seedmolindex, target, seed, top_score
    """
    rows = []

    layout = (layout or "nested").strip().lower()
    if layout == "reinvent4":
        if sim_thr is None:
            raise ValueError("--sim_thr is required when --layout reinvent4 (e.g. 0.4 or 0.6)")
        jobs_root = os.path.join(root_dir, "jobs")
        if not os.path.isdir(jobs_root):
            raise FileNotFoundError(f"No jobs/ folder under {root_dir}")
        sim_folder = sim_thr_to_folder(sim_thr)
        for target in targets:
            sim_path = os.path.join(jobs_root, target, sim_folder)
            if not os.path.isdir(sim_path):
                continue
            for mol_path in sorted(glob.glob(os.path.join(sim_path, "mol*"))):
                if not os.path.isdir(mol_path):
                    continue
                seedmolindex = int(os.path.basename(mol_path).replace("mol", ""))
                for job_path in sorted(glob.glob(os.path.join(mol_path, "*"))):
                    if not os.path.isdir(job_path):
                        continue
                    m = LEAD_JOB_RE.match(os.path.basename(job_path))
                    if not m:
                        continue
                    exp_num = HP_TO_EXP.get(m.group("hp"))
                    if exp_num is None:
                        continue
                    seed_num = int(m.group("seed"))
                    csv_path = resolve_seed_csv(job_path, molecules_csv_name)
                    if csv_path is None:
                        print(f"[SKIP] missing '{molecules_csv_name}' in {job_path}")
                        continue
                    try:
                        top_score = compute_top_score(
                            csv_path,
                            target,
                            seedmolindex,
                            ranges,
                            colmap,
                        )
                    except Exception as e:
                        print(f"[WARN] {csv_path}: {e}")
                        continue
                    rows.append(
                        {
                            "algorithm": algorithm_name,
                            "exp": exp_num,
                            "target": target,
                            "seedmolindex": seedmolindex,
                            "seed": seed_num,
                            "top_score": top_score,
                        }
                    )
    elif layout == "flat":
        exp_iter = [(int(flat_exp_index), root_dir)]
        for exp_num, exp_path in exp_iter:
            _walk_nested_flat(
                rows, exp_num, exp_path, targets, sim_thr_dir, start_mol_folder_base,
                molecules_csv_name, algorithm_name, ranges, colmap,
            )
    elif layout == "nested":
        exp_dirs = sorted(glob.glob(os.path.join(root_dir, "exp-*")))
        if not exp_dirs:
            raise FileNotFoundError(f"No exp-* folders under {root_dir}")
        exp_iter = [(suffix_int(os.path.basename(p)), p) for p in exp_dirs]
        for exp_num, exp_path in exp_iter:
            _walk_nested_flat(
                rows, exp_num, exp_path, targets, sim_thr_dir, start_mol_folder_base,
                molecules_csv_name, algorithm_name, ranges, colmap,
            )
    else:
        raise ValueError(f"Unknown layout '{layout}' (use 'nested', 'flat', or 'reinvent4')")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["exp", "target", "seedmolindex", "seed"]
        ).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


# ------------------------
# CLI
# ------------------------
def main():
    p = argparse.ArgumentParser(
        description="Flatten start_mol-idx hierarchy to top hit docking score"
    )
    p.add_argument("--root_dir", default=None, help="Root directory (default: cwd)")
    p.add_argument("--algorithm_name", required=True)
    p.add_argument("--out_csv", required=True)

    # required now (per your note)
    p.add_argument("--molecules_csv_name", required=True, help="CSV filename inside each seed-* folder")

    p.add_argument("--targets", nargs="*", default=["parp1", "fa7", "5ht1b", "jak2", "braf"])

    p.add_argument(
        "--layout",
        choices=("nested", "flat", "reinvent4"),
        default="nested",
        help="nested: root/exp-*/<target>/... (default). flat: root/<target>/... "
        "reinvent4: root/jobs/<target>/sim04|sim06/mol*/<job>/molecules.csv.",
    )
    p.add_argument(
        "--sim_thr",
        type=float,
        default=None,
        help="Similarity threshold for --layout reinvent4 (0.4 -> sim04, 0.6 -> sim06). "
        "Also sets hit-filter similarity min unless --similarity_min is given.",
    )
    p.add_argument(
        "--similarity_min",
        type=float,
        default=None,
        help="Minimum similarity for hit filtering (default: RANGES or --sim_thr).",
    )
    p.add_argument(
        "--flat_exp_index",
        type=int,
        default=0,
        help="Value for the 'exp' column when using --layout flat.",
    )
    p.add_argument(
        "--sim_thr_dir",
        default="sim_thr-0.6",
        help="Folder name under each target (default: sim_thr-0.6).",
    )
    p.add_argument(
        "--start_mol_folder_base",
        type=int,
        default=1,
        help="Integer B so seedmolindex = <start_mol_idx-N suffix> - B. "
        "Use 1 for folders start_mol_idx-1,2,3 (default). Use 0 for start_mol_idx-0,1,2.",
    )

    # column names are configurable
    p.add_argument("--qed_col", required=True, help="Column name for QED in molecules.csv")
    p.add_argument("--sa_col", required=True, help="Column name for SA in molecules.csv")
    p.add_argument("--similarity_col", required=True, help="Column name for similarity in molecules.csv")

    # docking col is optional: if not provided, it tries <target> then 'ds'
    p.add_argument("--docking_col", default=None, help="Docking column name (optional)")

    args = p.parse_args()
    root_dir = os.getcwd() if args.root_dir is None else os.path.abspath(os.path.expanduser(args.root_dir))

    colmap = {
        "qed": args.qed_col,
        "sa": args.sa_col,
        "similarity": args.similarity_col,
        "docking": args.docking_col,
    }

    ranges = dict(RANGES)
    ranges["similarity"] = dict(RANGES["similarity"])
    if args.similarity_min is not None:
        ranges["similarity"]["min"] = float(args.similarity_min)
    elif args.sim_thr is not None:
        ranges["similarity"]["min"] = float(args.sim_thr)

    df = flatten_top_scores(
        root_dir=root_dir,
        algorithm_name=args.algorithm_name,
        out_csv=args.out_csv,
        ranges=ranges,
        targets=args.targets,
        molecules_csv_name=args.molecules_csv_name,
        colmap=colmap,
        layout=args.layout,
        flat_exp_index=args.flat_exp_index,
        sim_thr_dir=args.sim_thr_dir,
        start_mol_folder_base=args.start_mol_folder_base,
        sim_thr=args.sim_thr,
    )

    print(f"\nWrote {len(df)} rows → {args.out_csv}")
    if len(df) > 0:
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()



