#!/usr/bin/env python3
"""
Flatten hierarchy (genetic-genmol/genmol_hit/hparam_search_spec):

  root_dir/
    exp-*/
      <target>_<antitarget>/   or   <target>-<antitarget>/
        seed-*/
          results.csv

Per seed:
  - read seed CSV
  - (optional) de-duplicate by canonical SMILES (if --smiles_col provided)
  - filter hit rows by qed/sa/dock1 ranges (inclusive)
  - eligible = hit & dock2_notna
  - delta = max(dock1 - dock2, 0)   # clip negatives to 0
  - take top_k deltas, pad missing with fallback delta for that (target, anti_target)
  - store mean(top_k padded deltas)

Output CSV columns:
  algorithm, exp, target, anti_target, seed, mean_top10_delta

CSV schema (default columns):
  molecule,docking_score,qed_score,sa_score,antitarget_score,truncated

Note: This implementation intentionally avoids a hard dependency on pandas.
"""

from __future__ import annotations

import os
import glob
import argparse
import json
import csv
import math
import heapq
from typing import Any


# ------------------------
# helpers
# ------------------------
def suffix_int(name: str) -> int:
    """Extract integer suffix from exp-3 / seed-12."""
    try:
        return int(name.split("-")[-1])
    except Exception as e:
        raise ValueError(f"Cannot parse numeric suffix from '{name}'") from e


def parse_target_antitarget(folder_name: str) -> tuple[str, str]:
    """
    Accept:
      - <target>_<antitarget>
      - <target>-<antitarget>
      - spec.<target>_<antitarget>   (prefix ignored)
    """
    base = folder_name.strip()
    lower = base.lower()
    if lower.startswith("spec.") and len(base) > 5:
        base = base[5:]

    if "_" in base:
        target, ant = base.split("_", 1)
    elif "-" in base:
        target, ant = base.split("-", 1)
    else:
        raise ValueError(
            f"Expected '<target>_<antitarget>' or '<target>-<antitarget>', got '{folder_name}'"
        )

    target = target.strip().lower()
    ant = ant.strip().lower()
    if not target or not ant:
        raise ValueError(f"Invalid folder name: '{folder_name}'")
    return target, ant


def json_dict_or_none(s: str | None) -> dict | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return json.loads(s)


def resolve_dock_column(entity_name: str, overrides: dict[str, str] | None, default_col: str) -> str:
    """
    entity_name: target_name or antitarget_name
    overrides: mapping entity_name -> column_name
    default_col: fallback column
    """
    if not overrides:
        return default_col.strip().lower()
    lowered = {str(k).strip().lower(): str(v).strip().lower() for k, v in overrides.items()}
    return lowered.get(entity_name.strip().lower(), default_col.strip().lower())


def resolve_seed_csv(seed_dir: str, molecules_csv_name: str | None) -> str | None:
    """
    Priority:
      1) molecules_csv_name if provided AND exists
      2) exactly one *.csv in seed_dir
      3) None (ambiguous or missing)
    """
    if molecules_csv_name:
        preferred = os.path.join(seed_dir, molecules_csv_name)
        if os.path.isfile(preferred):
            return preferred

    csvs = sorted(glob.glob(os.path.join(seed_dir, "*.csv")))
    if len(csvs) == 1:
        return csvs[0]

    return None


def sniff_csv_dialect(path: str) -> csv.Dialect:
    with open(path, "r", newline="") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
    except Exception:
        return csv.excel


def to_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v):
        return None
    return v


# ------------------------
# optional canonical SMILES de-dup
# ------------------------
def canonicalize_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem  # type: ignore
    except Exception as e:
        raise ImportError(
            "RDKit is required for --smiles_col de-dup but could not be imported."
        ) from e

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit failed to parse SMILES")
    return Chem.MolToSmiles(mol, canonical=True)


def compute_mean_topk_delta(
    csv_path: str,
    *,
    qed_col: str,
    sa_col: str,
    dock1_col: str,
    dock2_col: str,
    ranges: dict,
    top_k: int,
    fill_delta: float,
    smiles_col: str | None = None,
    print_delta_ge: float | None = None,
    print_delta_limit: int = 200,
    context: str | None = None,
    warn_invalid_rows: bool = True,
    debug: bool = False,
) -> float:
    """
    Same math as the other flatten_deltas scripts:
      dedup (optional) -> hit filter -> eligible -> delta=max(d1-d2,0) -> top_k -> pad -> mean
    """
    qed_col_lc = qed_col.strip().lower()
    sa_col_lc = sa_col.strip().lower()
    dock1_col_lc = dock1_col.strip().lower()
    dock2_col_lc = dock2_col.strip().lower()
    smiles_col_lc = smiles_col.strip().lower() if smiles_col else None

    dialect = sniff_csv_dialect(csv_path)

    n_total = 0
    n_invalid_hit = 0
    n_hit = 0
    n_eligible = 0
    n_dedup_dropped = 0
    n_dedup_bad = 0
    n_printed = 0

    seen_can: set[str] | None = set() if smiles_col_lc is not None else None
    heap: list[float] = []  # min-heap holding top_k largest deltas
    heap_sum = 0.0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            return float(fill_delta)
        field_map = {name.strip().lower(): name for name in reader.fieldnames}

        missing = {qed_col_lc, sa_col_lc, dock1_col_lc, dock2_col_lc} - set(field_map.keys())
        if smiles_col_lc is not None:
            missing |= ({smiles_col_lc} - set(field_map.keys()))
        if missing:
            raise ValueError(
                f"Missing columns {sorted(missing)} in {csv_path}. Available: {list(field_map.keys())}"
            )

        for row in reader:
            n_total += 1

            if smiles_col_lc is not None and seen_can is not None:
                smi = (row.get(field_map[smiles_col_lc]) or "").strip()
                if smi and smi.lower() not in {"nan", "none"}:
                    try:
                        can = canonicalize_smiles(smi)
                    except Exception:
                        n_dedup_bad += 1
                    else:
                        if can in seen_can:
                            n_dedup_dropped += 1
                            continue
                        seen_can.add(can)

            qed = to_float(row.get(field_map[qed_col_lc]))
            sa = to_float(row.get(field_map[sa_col_lc]))
            d1 = to_float(row.get(field_map[dock1_col_lc]))
            d2 = to_float(row.get(field_map[dock2_col_lc]))

            valid_hit = (qed is not None) and (sa is not None) and (d1 is not None)
            if not valid_hit:
                n_invalid_hit += 1
                continue

            hit = (
                (ranges["qed"]["min"] <= qed <= ranges["qed"]["max"])
                and (ranges["sa"]["min"] <= sa <= ranges["sa"]["max"])
                and (ranges["dock1"]["min"] <= d1 <= ranges["dock1"]["max"])
            )
            if not hit:
                continue
            n_hit += 1

            if d2 is None:
                continue
            n_eligible += 1

            delta = d1 - d2
            if delta < 0:
                delta = fill_delta

            # Print only rows that satisfy constraints (hit + dock2 notna == eligible)
            # and have large delta.
            if (
                print_delta_ge is not None
                and delta >= float(print_delta_ge)
                and n_printed < int(print_delta_limit)
            ):
                prefix = f"[DELTA>= {float(print_delta_ge):g}]"
                ctx = f" {context}" if context else ""
                mol = row.get(field_map.get(smiles_col_lc, ""), "") if smiles_col_lc else ""
                # If no smiles_col passed, try common column names for convenience.
                if not mol:
                    for cand in ("molecule", "mol", "smiles"):
                        if cand in field_map:
                            mol = row.get(field_map[cand], "")
                            break
                mol = (mol or "").strip()
                print(
                    f"{prefix}{ctx} csv={csv_path} "
                    f"oracle_row={n_total} "
                    f"delta={delta:.6g} "
                    f"{dock1_col_lc}={d1:.6g} {dock2_col_lc}={d2:.6g} "
                    f"{qed_col_lc}={qed:.6g} {sa_col_lc}={sa:.6g} "
                    f"mol={mol}"
                )
                n_printed += 1

            if len(heap) < top_k:
                heapq.heappush(heap, float(delta))
                heap_sum += float(delta)
            else:
                # keep only the top_k largest deltas
                if delta > heap[0]:
                    heap_sum -= heap[0]
                    heapq.heapreplace(heap, float(delta))
                    heap_sum += float(delta)

    if warn_invalid_rows and n_invalid_hit > 0:
        print(f"[WARN] {os.path.basename(csv_path)}: {n_invalid_hit} invalid rows (NaN in {qed_col_lc}/{sa_col_lc}/{dock1_col_lc})")

    if debug:
        print(
            f"[DEBUG] {os.path.basename(csv_path)}: total={n_total}, hit={n_hit}, eligible={n_eligible}, "
            f"topk_found={len(heap)}, fill_delta={fill_delta}"
        )
        if smiles_col_lc is not None:
            print(f"[DEBUG] {os.path.basename(csv_path)}: dedup dropped={n_dedup_dropped}, bad_smiles={n_dedup_bad}")

    if len(heap) == 0:
        return float(fill_delta)

    if len(heap) < top_k:
        heap_sum += float(fill_delta) * float(top_k - len(heap))

    return float(heap_sum / float(top_k))


def flatten_exp_target_antitarget(
    *,
    root_dir: str,
    algorithm_name: str,
    out_csv: str,
    molecules_csv_name: str | None,
    qed_col: str,
    sa_col: str,
    smiles_col: str | None,
    target_col_overrides: dict[str, str] | None,
    anti_col_overrides: dict[str, str] | None,
    default_target_col: str = "docking_score",
    default_anti_col: str = "antitarget_score",
    ranges: dict | None = None,
    top_k: int = 10,
    fill_delta_by_pair: dict[tuple[str, str], float] | None = None,
    exp_glob: str = "exp-*",
    seed_glob: str = "seed-*",
    warn_invalid_rows: bool = True,
    debug: bool = False,
    print_delta_ge: float | None = None,
    print_delta_limit: int = 200,
    layout: str = "nested",
    flat_exp_index: int = 0,
) -> None:
    if ranges is None:
        ranges = {
            "qed": {"min": 0.4, "max": 1.0},
            "sa": {"min": 1, "max": 4.0},
            "dock1": {"min": 10.67, "max": 20.0},
        }
    if fill_delta_by_pair is None:
        fill_delta_by_pair = {
            ("6nzp", "7uyt"): 2.21,
            ("6nzp", "5ut5"): 1.73,
            ("6nzp", "7uyw"): 2.04,
            ("6nzp", "5khw"): 1.97,
            ("6nzp", "4l00"): 2.47,
        }
    fill_delta_by_pair = {
        (a.strip().lower(), b.strip().lower()): float(v)
        for (a, b), v in fill_delta_by_pair.items()
    }

    rows: list[dict[str, Any]] = []

    layout = (layout or "nested").strip().lower()
    if layout == "flat":
        exp_iter: list[tuple[int, str]] = [(int(flat_exp_index), root_dir)]
    elif layout == "nested":
        exp_dirs = sorted(glob.glob(os.path.join(root_dir, exp_glob)))
        if not exp_dirs:
            raise FileNotFoundError(f"No {exp_glob} folders under {root_dir}")
        exp_iter = []
        for exp_path in exp_dirs:
            exp_base = os.path.basename(exp_path)
            try:
                exp_num = suffix_int(exp_base)
            except Exception:
                continue
            exp_iter.append((exp_num, exp_path))
    else:
        raise ValueError(f"Unknown layout '{layout}' (use 'nested' or 'flat')")

    for exp_num, exp_path in exp_iter:
        for pair_path in sorted([p for p in glob.glob(os.path.join(exp_path, "*")) if os.path.isdir(p)]):
            pair_folder = os.path.basename(pair_path)
            try:
                target_name, ant_name = parse_target_antitarget(pair_folder)
            except Exception:
                continue

            dock1_col = resolve_dock_column(target_name, target_col_overrides, default_target_col)
            dock2_col = resolve_dock_column(ant_name, anti_col_overrides, default_anti_col)
            fill_delta = float(fill_delta_by_pair.get((target_name, ant_name), 0.0))

            for seed_path in sorted(glob.glob(os.path.join(pair_path, seed_glob))):
                seed_base = os.path.basename(seed_path)
                try:
                    seed_num = suffix_int(seed_base)
                except Exception:
                    continue

                csv_path = resolve_seed_csv(seed_path, molecules_csv_name)
                if csv_path is None:
                    continue

                try:
                    mean_top10 = compute_mean_topk_delta(
                        csv_path,
                        qed_col=qed_col,
                        sa_col=sa_col,
                        dock1_col=dock1_col,
                        dock2_col=dock2_col,
                        ranges=ranges,
                        top_k=top_k,
                        fill_delta=fill_delta,
                        smiles_col=smiles_col,
                        print_delta_ge=print_delta_ge,
                        print_delta_limit=print_delta_limit,
                        context=f"[exp={exp_num} pair={target_name}_{ant_name} seed={seed_num}]",
                        warn_invalid_rows=warn_invalid_rows,
                        debug=debug,
                    )
                except Exception as e:
                    print(f"[WARN] {csv_path}: {e}")
                    continue

                rows.append(
                    {
                        "algorithm": algorithm_name,
                        "exp": exp_num,
                        "target": target_name,
                        "anti_target": ant_name,
                        "seed": seed_num,
                        "mean_top10_delta": mean_top10,
                    }
                )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["algorithm", "exp", "target", "anti_target", "seed", "mean_top10_delta"],
        )
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["exp"], x["target"], x["anti_target"], x["seed"])):
            w.writerow(r)

    print(f"\nWrote {len(rows)} rows → {out_csv}")


def main():
    p = argparse.ArgumentParser(
        description="Flatten genmol_hit hparam_search_spec exp/<target>_<antitarget>/seed hierarchy to mean(top-k deltas)"
    )
    p.add_argument("--root_dir", default=None, help="Root directory (default: cwd)")
    p.add_argument("--algorithm_name", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--molecules_csv_name", default="results.csv")

    # Column config (defaults match results.csv)
    p.add_argument("--qed_col", default="qed_score")
    p.add_argument("--sa_col", default="sa_score")
    p.add_argument("--smiles_col", default=None, help="Enable canonical SMILES de-dup (requires RDKit).")
    p.add_argument("--dock1_col", default="docking_score")
    p.add_argument("--dock2_col", default="antitarget_score")

    p.add_argument("--target_col_overrides", default=None)
    p.add_argument("--anti_col_overrides", default=None)

    # Ranges
    p.add_argument("--qed_min", type=float, default=0.4)
    p.add_argument("--qed_max", type=float, default=1.0)
    p.add_argument("--sa_min", type=float, default=1.0)
    p.add_argument("--sa_max", type=float, default=4.0)
    p.add_argument("--dock1_min", type=float, default=10.67)
    p.add_argument("--dock1_max", type=float, default=20.0)

    # Top-k
    p.add_argument("--top_k", type=int, default=10)

    p.add_argument(
        "--layout",
        choices=("nested", "flat"),
        default="nested",
        help="nested: root/exp-*/<pair>/seed-* (default). flat: root/<pair>/seed-* (e.g. spec_best).",
    )
    p.add_argument(
        "--exp_glob",
        default="exp-*",
        help="Experiment folder glob under root (nested layout only).",
    )
    p.add_argument(
        "--flat_exp_index",
        type=int,
        default=0,
        help="Value for the 'exp' column when using --layout flat.",
    )

    p.add_argument("--no_warn_invalid_rows", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--print_delta_ge",
        type=float,
        default=4.0,
        help="If set, print eligible rows with delta >= threshold (default: 4.0). Use 0 to print all eligible.",
    )
    p.add_argument(
        "--print_delta_limit",
        type=int,
        default=200,
        help="Max number of rows to print per seed CSV when --print_delta_ge is enabled (default: 200).",
    )

    args = p.parse_args()
    root_dir = os.getcwd() if args.root_dir is None else os.path.abspath(os.path.expanduser(args.root_dir))

    ranges = {
        "qed": {"min": args.qed_min, "max": args.qed_max},
        "sa": {"min": args.sa_min, "max": args.sa_max},
        "dock1": {"min": args.dock1_min, "max": args.dock1_max},
    }

    flatten_exp_target_antitarget(
        root_dir=root_dir,
        algorithm_name=args.algorithm_name,
        out_csv=args.out_csv,
        molecules_csv_name=args.molecules_csv_name,
        qed_col=args.qed_col,
        sa_col=args.sa_col,
        smiles_col=args.smiles_col,
        target_col_overrides=json_dict_or_none(args.target_col_overrides),
        anti_col_overrides=json_dict_or_none(args.anti_col_overrides),
        default_target_col=args.dock1_col,
        default_anti_col=args.dock2_col,
        ranges=ranges,
        top_k=args.top_k,
        exp_glob=args.exp_glob,
        warn_invalid_rows=(not args.no_warn_invalid_rows),
        debug=args.debug,
        print_delta_ge=args.print_delta_ge,
        print_delta_limit=args.print_delta_limit,
        layout=args.layout,
        flat_exp_index=args.flat_exp_index,
    )


if __name__ == "__main__":
    main()
