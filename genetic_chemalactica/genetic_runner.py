import argparse
import os
import sys
import yaml
import subprocess
import itertools
import logging
import time
import copy
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

_gc_root = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(_gc_root)
sys.path.insert(0, repo_root)

from benchmark.experiment_utils import get_job_dir, get_log_dir
from genetic_chemalactica.utils.tasks import validate_task_name, validate_reward_type


def to_nested_dict(orig_dict):
    """Convert nested dict to flat dict with dot-notation keys"""
    nested_dict = {}
    for key, value in orig_dict.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                nested_dict[f"{key}.{nested_key}"] = nested_value
        else:
            nested_dict[key] = value
    return nested_dict

def prepare_hparam_config(original_config_dict, hparam_config_path):
    """Prepare hyperparameter configs from a hparam config file"""
    with open(hparam_config_path, "r") as f:
        hparam_config = yaml.safe_load(f)

    # Convert nested dict to flat dict with dot-notation keys
    nested_hparam_dict = to_nested_dict(hparam_config)
    
    # Get all keys and their values
    keys = list(nested_hparam_dict.keys())
    values = [nested_hparam_dict[key] for key in keys]
    
    # Generate all possible combinations using Cartesian product
    config_dicts = []
    for combination in itertools.product(*values):
        config_dict = copy.deepcopy(original_config_dict)
        for key, value in zip(keys, combination):
            # Handle nested keys (e.g., "genetic.pool_size" -> config_dict["genetic"]["pool_size"])
            if "." in key:
                parts = key.split(".")
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value
        config_dicts.append(config_dict)
    
    return config_dicts


def _extract_hparam_combo(config_dict: dict) -> tuple[int | None, int | None, int | None]:
    g = (config_dict or {}).get("genetic", {}) if isinstance(config_dict, dict) else {}
    return (g.get("pool_size"), g.get("num_prompts"), g.get("num_similars"))


def _load_done_hparam_combos_from_dir(done_dir: str) -> set[tuple[int | None, int | None, int | None]]:
    """
    Read config-*.yaml files in an existing results directory (exp-*/spec.*/config-*.yaml)
    and return the set of (pool_size, num_prompts, num_similars) combos already run.
    """
    combos: set[tuple[int | None, int | None, int | None]] = set()
    if not done_dir:
        return combos
    cfg_paths = glob(os.path.join(done_dir, "exp-*", "spec.*", "config-*.yaml"))
    for p in cfg_paths:
        try:
            with open(p, "r") as f:
                cfg = yaml.safe_load(f)
            combos.add(_extract_hparam_combo(cfg))
        except Exception:
            continue
    return combos
    
def run_genetic(cfg_path):
    """Run genetic/run.py for one config.

    GPU selection is left to the scheduler (e.g. SLURM ``CUDA_VISIBLE_DEVICES``).
    This runner does not rewrite the config.
    """
    cfg_abs = os.path.abspath(cfg_path)

    proj = os.environ.get("PROJECT_ROOT", repo_root)
    run_py = os.path.abspath(os.path.join(proj, "genetic_chemalactica", "genetic", "run.py"))
    workdir = os.path.abspath(os.path.join(proj, "genetic_chemalactica"))

    genetic_command = [
        sys.executable,
        run_py,
        "--config_file",
        cfg_abs,
    ]

    env = os.environ.copy()
    # DOCKING_VINA_URL may be set by the parent job environment; run.py also sets it from config when present.

    genetic_process = subprocess.Popen(
        genetic_command,
        env=env,
        cwd=workdir,
    )
    genetic_process.wait()
    return genetic_process.returncode


def _infer_n_gpus(explicit_n_gpus: int | None) -> int:
    """Infer number of visible GPUs from SLURM/CUDA_VISIBLE_DEVICES."""
    if explicit_n_gpus is not None:
        try:
            n = int(explicit_n_gpus)
            return max(1, n)
        except Exception:
            return 1
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return 1
    # CUDA_VISIBLE_DEVICES can be "0", "0,1", or UUIDs; we only need the count.
    parts = [p for p in cvd.split(",") if p.strip() != ""]
    return max(1, len(parts))


def _round_robin_device(i: int, n_gpus: int) -> str:
    return f"cuda:{i % max(1, n_gpus)}"


def run_genetics(cfg_paths, max_workers=None):
    """Run multiple seed and oracle pairs in parallel in a single job"""
    job_start = time.time()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Limit parallel workers to avoid excessive memory usage
    if max_workers is None:
        max_workers = min(len(cfg_paths), 20)  # Default to max 20 parallel workers
    
    logging.info(f"Starting {len(cfg_paths)} genetic runs with {max_workers} parallel workers")
    
    def run_single_genetic(cfg_path):
        """Helper function to run a single seed and oracle pair"""
        try:
            return_code = run_genetic(cfg_path)
            if return_code == 0:
                return cfg_path, None
            else:
                error_msg = f"Exit code: {return_code}"
                logging.error(f"{cfg_path}: {error_msg}")
                return cfg_path, error_msg
        except Exception as e:
            error_msg = str(e)
            logging.error(f"{cfg_path}: {error_msg}")
            return cfg_path, error_msg
    
    # Run genetics in parallel using ThreadPoolExecutor with limited workers
    delay = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cfg = {}
        for i, cfg_path in enumerate(cfg_paths):
            if i > 0:
                time.sleep(delay)
            future = executor.submit(run_single_genetic, cfg_path)
            future_to_cfg[future] = cfg_path
        
        results = []
        for future in as_completed(future_to_cfg):
            cfg_path, error = future.result()
            results.append((cfg_path, error))
    
    # Report summary
    job_time = time.time() - job_start
    successful = sum(1 for _, error in results if error is None)
    failed = len(results) - successful
    
    logging.info(f"Complete: {successful} successful, {failed} failed ({job_time/60:.1f}min)")
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=False,
        type=str
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        required=False,
        type=int
    )
    parser.add_argument(
        "--task_name",
        nargs="+",
        required=False,
        type=str,
        help="Task name(s), e.g. hit.parp1, lead.parp1_04_0, spec.6nzp_7uyt.",
    )
    parser.add_argument(
        "--vina_url",
        required=False,
        type=str,
        help="Optional docking HTTP service URL for parallel runs (default: in-process DockingOracle)",
    )
    parser.add_argument(
        "--max_oracle_calls",
        required=True,
        type=int
    )
    parser.add_argument(
        "--reward_type",
        required=False,
        type=str,
        default="hit",
        help="Reward type for hit.<target> docking tasks: hit (default), max, or geam. Ignored for lead/spec tasks."
    )
    parser.add_argument(
        "--hparam_config",
        type=str,
        required=False,
        default=None
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        required=False,
        default=15,
        help="Maximum parallel genetic subprocesses (GPU visibility from SLURM / environment; each child uses cuda:0).",
    )
    parser.add_argument(
        "--search_range",
        nargs="+",
        type=int,
        required=False,
        default=None,
        help="Slice [start:end) of the hparam list after --skip_completed_dir (if set), else of the full Cartesian product.",
    )
    parser.add_argument(
        "--skip_completed_dir",
        type=str,
        required=False,
        default=None,
        help="If set (and --hparam_config is provided), skip hparam combos already present under this results dir (exp-*/spec.*/config-*.yaml).",
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        required=False,
        default=None,
        help="Optional number of GPUs to round-robin over (default: inferred from CUDA_VISIBLE_DEVICES).",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        required=False,
        default=None,
        help="Existing seed log dir (…/lead.<task>/seed-K) or task dir. Reuse 0_mols.csv instead of a new exp.",
    )
    args = parser.parse_args()

    # Validate required arguments
    if args.task_name is None:
        raise ValueError("--task_name is required")
    if args.seeds is None:
        raise ValueError("--seeds is required")
    for task in args.task_name:
        validate_task_name(task)
        validate_reward_type(task, args.reward_type)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    job_dir = get_job_dir(args.hparam_config is not None, cat=f"genetic")
    n_gpus = _infer_n_gpus(args.n_gpus)
    logging.info(
        "GPU selection: inherited from environment (e.g. SLURM CUDA_VISIBLE_DEVICES). "
        "device assignment: always round_robin (n_gpus=%s, CUDA_VISIBLE_DEVICES=%s)",
        n_gpus,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )

    # Load base config
    with open(args.config_file, "r") as f:
        orig_config_dict = yaml.safe_load(f)
    if args.hparam_config is not None:
        config_dicts = prepare_hparam_config(orig_config_dict, args.hparam_config)
    else:
        config_dicts = [orig_config_dict]

    
    all_cfg_paths = []

    if args.hparam_config is not None and args.skip_completed_dir:
        done = _load_done_hparam_combos_from_dir(args.skip_completed_dir)
        before = len(config_dicts)
        config_dicts = [c for c in config_dicts if _extract_hparam_combo(c) not in done]
        logging.info(
            "Skipping completed hparam combos from %s: %s -> %s configs",
            args.skip_completed_dir,
            before,
            len(config_dicts),
        )
    if args.search_range is not None:
        a, b = int(args.search_range[0]), int(args.search_range[1])
        before_slice = len(config_dicts)
        config_dicts = config_dicts[a:b]
        logging.info(
            "search_range [%s:%s]: %s -> %s configs",
            a,
            b,
            before_slice,
            len(config_dicts),
        )
    device_counter = 0
    resume_root = os.path.abspath(args.resume_dir) if args.resume_dir else None
    if resume_root and not os.path.isdir(resume_root):
        raise ValueError(f"--resume_dir does not exist: {resume_root}")

    for config_dict in config_dicts:
        if resume_root:
            log_dir = resume_root
            if os.path.basename(resume_root).startswith("seed-"):
                log_dir = os.path.dirname(resume_root)
            if os.path.basename(log_dir).startswith("lead.") or os.path.basename(log_dir).startswith("hit.") or os.path.basename(log_dir).startswith("spec."):
                log_dir = os.path.dirname(log_dir)
        else:
            task_category = args.task_name[0].split(".")[0]
            log_dir = get_log_dir(
                method="genetic",
                model_name=task_category,
                exp_name="exp",
                suffix="-hparam" if args.hparam_config else "",
            )
        os.makedirs(log_dir, exist_ok=True)

        for task in args.task_name:
            task_log_dir = os.path.join(log_dir, task)
            os.makedirs(task_log_dir, exist_ok=True)

            for seed in args.seeds:
                seed_log_dir = os.path.join(task_log_dir, f"seed-{seed}")
                if resume_root:
                    base = os.path.basename(resume_root)
                    if base == f"seed-{seed}":
                        seed_log_dir = resume_root
                    elif base.startswith("seed-") and len(args.seeds) == 1:
                        seed_log_dir = resume_root
                    elif os.path.isdir(os.path.join(resume_root, f"seed-{seed}")):
                        seed_log_dir = os.path.join(resume_root, f"seed-{seed}")
                    elif os.path.isdir(os.path.join(resume_root, task, f"seed-{seed}")):
                        seed_log_dir = os.path.join(resume_root, task, f"seed-{seed}")
                    csv_path = os.path.join(seed_log_dir, "0_mols.csv")
                    if not os.path.isfile(csv_path):
                        raise FileNotFoundError(f"resume CSV not found: {csv_path}")
                os.makedirs(seed_log_dir, exist_ok=True)

                seed_config_dict = copy.deepcopy(config_dict)

                if "oracle" not in seed_config_dict:
                    seed_config_dict["oracle"] = {}

                seed_config_dict["seed"] = seed
                seed_config_dict["oracle"]["task_name"] = task
                seed_config_dict["oracle"]["max_calls"] = args.max_oracle_calls
                seed_config_dict["oracle"]["log_dir"] = seed_log_dir
                seed_config_dict["oracle"]["reward_type"] = args.reward_type
                if resume_root:
                    seed_config_dict["oracle"]["resume"] = True

                if args.vina_url is not None:
                    seed_config_dict["vina_url"] = args.vina_url
                seed_config_dict["device"] = _round_robin_device(device_counter, n_gpus)
                device_counter += 1

                cfg_file = os.path.join(task_log_dir, f"config-{seed}.yaml")
                with open(cfg_file, "w") as f:
                    yaml.safe_dump(seed_config_dict, f)
                all_cfg_paths.append(cfg_file)

    total_workers = args.max_workers
    logging.info(f"Running {len(all_cfg_paths)} jobs with up to {total_workers} parallel workers")
    
    n_failed = run_genetics(cfg_paths=all_cfg_paths, max_workers=total_workers)
    logging.info(f"Completed {len(all_cfg_paths)} jobs")
    sys.exit(1 if n_failed else 0)
