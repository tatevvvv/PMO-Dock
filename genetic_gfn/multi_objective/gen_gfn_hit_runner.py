import argparse
import os
import sys
import yaml
import subprocess
import itertools
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add repo root to path for benchmark imports
# This file is at: <REPO_ROOT>/GeneticGFN/multi_objective/gen_gfn_hit_runner.py
repo_root = os.environ.get("PROJECT_ROOT")
if not repo_root:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, repo_root)

from benchmark.experiment_utils import get_job_dir, get_log_dir

from benchmark.spec_tasks import list_antitarget_receptors

# Selectivity configuration: target -> list of anti-target docking receptors.
TARGET_TO_ANTI_TARGETS = {
    "6nzp": list_antitarget_receptors(),
}


def prepare_hparam_config(original_config_dict, hparam_config_path):
    with open(hparam_config_path, "r") as f:
        hparam_config = yaml.safe_load(f)

    # Get all keys and their values
    keys = list(hparam_config.keys())
    # Support both list (sweep) and scalar (fixed) values
    values = [hparam_config[key] if isinstance(hparam_config[key], list) else [hparam_config[key]] for key in keys]
    
    # Generate all possible combinations using Cartesian product
    config_dicts = []
    for combination in itertools.product(*values):
        config_dict = original_config_dict.copy()
        for key, value in zip(keys, combination):
            config_dict[key] = value
        config_dicts.append(config_dict)
    
    return config_dicts

def run_hit(cfg_path):
    """Run one GeneticGFN hit job; GPU visibility from scheduler / environment."""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    gfn_hit_command = [
        "python3",
        "-u",
        f"{os.environ['PROJECT_ROOT']}/genetic_gfn/multi_objective/run.py",
        "genetic_gfn",
        "--task", "simple",
        "--seed", str(cfg["seed"]),
        "--objectives", str(cfg["objectives"]),
        "--alpha_vector", str(cfg.get("alpha_vector", "1,1,1")),
        "--max_oracle_calls", str(cfg["max_oracle_calls"]),
        "--freq_log", str(cfg.get("freq_log", 100)),
        "--output_dir", str(cfg["output_dir"]),
        "--run_name", str(cfg["run_name"]),
        "--config_default", str(cfg["config_default_path"]),
    ]
    # Selectivity anti-target (required by run.py when objectives contain 6nzp)
    anti_target = (cfg.get("anti_target") or "").strip()
    if anti_target:
        gfn_hit_command += ["--anti_target", anti_target]
    # Pass through CPU parallelism (joblib pool size). run.py will map this to config["num_jobs"].
    if cfg.get("n_jobs") is not None:
        try:
            gfn_hit_command += ["--n_jobs", str(int(cfg["n_jobs"]))]
        except Exception:
            pass
    # Pass oracle_url explicitly (GeneticGFN now reads it via args/config, not env vars)
    if cfg.get("oracle_url"):
        gfn_hit_command += ["--oracle_url", str(cfg["oracle_url"])]
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Pin each child to one physical GPU (runner always uses cuda:0 inside the process).
    gpu_id = cfg.get("gpu_id")
    if gpu_id is not None and str(gpu_id).strip() != "":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    multi_obj_dir = os.path.join(os.environ["PROJECT_ROOT"], "genetic_gfn", "multi_objective")
    # Log stdout/stderr to per-run files so users don't need to scan terminal output.
    # cfg_path is: <seed_log_dir>/run_config.yaml
    run_dir = os.path.dirname(os.path.abspath(cfg_path))
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    run_name_safe = str(cfg.get("run_name") or "run").replace(os.sep, "_")
    stdout_path = os.path.join(logs_dir, f"{run_name_safe}.stdout.log")
    stderr_path = os.path.join(logs_dir, f"{run_name_safe}.stderr.log")

    with open(stdout_path, "w", buffering=1) as stdout_f, open(stderr_path, "w", buffering=1) as stderr_f:
        stdout_f.write(f"CWD: {multi_obj_dir}\n")
        stdout_f.write(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','')}\n")
        stdout_f.write("CMD: " + " ".join(gfn_hit_command) + "\n\n")
        stdout_f.flush()

        # Prefer the env interpreter so we don't accidentally use system python3.
        gfn_hit_command[0] = sys.executable
        gfn_hit_process = subprocess.Popen(
            gfn_hit_command,
            env=env,
            cwd=multi_obj_dir,
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
        )
        gfn_hit_process.wait()

    return gfn_hit_process.returncode

def run_hits(cfg_paths, max_workers=None):
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
        max_workers = min(len(cfg_paths), 5)  # Default to max 5 parallel workers
    
    logging.info(f"Starting {len(cfg_paths)} hits with {max_workers} parallel workers")
    
    def run_single_hit(cfg_path):
        """Helper function to run a single seed and oracle pair"""
        try:
            return_code = run_hit(cfg_path)
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
    
    # Run hits in parallel using ThreadPoolExecutor with limited workers
    delay = 10
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cfg = {}
        for i, cfg_path in enumerate(cfg_paths):
            if i > 0:
                time.sleep(delay)
            future = executor.submit(run_single_hit, cfg_path)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        type=str
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        required=True,
        type=int
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
        type=str
    )
    # Backwards-compatible alias (deprecated)
    parser.add_argument(
        "--oracle_name",
        nargs="+",
        required=False,
        type=str,
        help="DEPRECATED: use --targets"
    )
    parser.add_argument(
        "--oracle_url",
        required=False,
        type=str
    )
    parser.add_argument(
        "--max_oracle_calls",
        required=True,
        type=int
    )
    parser.add_argument(
        "--n_jobs",
        required=False,
        default=-1,
        type=int,
        help="CPU workers passed to GeneticGFN/multi_objective/run.py --n_jobs. -1 means 'do not override YAML num_jobs'."
    )
    parser.add_argument(
        "--alpha_vector",
        required=False,
        default="1,1,1",
        type=str
    )
    parser.add_argument(
        "--objectives_prefix",
        required=False,
        default="qed,sa",
        type=str
    )
    parser.add_argument(
        "--freq_log",
        required=False,
        default=100,
        type=int
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
        help="Maximum parallel subprocesses (GPU from SLURM / environment; each run uses cuda:0).",
    )
    parser.add_argument(
        "--anti_targets",
        nargs="+",
        required=False,
        default=None,
        help="If set with --targets 6nzp, only these antitargets are run (default: all configured).",
    )
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        type=int,
        required=False,
        default=None,
        help="Physical GPU ids to round-robin across parallel runs (sets CUDA_VISIBLE_DEVICES per child).",
    )
    parser.add_argument(
        "--search_range",
        type=int,
        nargs="+",
        required=False,
        default=None
    )
    args = parser.parse_args()

    targets = args.targets if args.targets else args.oracle_name
    if not targets:
        raise ValueError("--targets is required")
    if args.anti_targets:
        for t in targets:
            if str(t) in TARGET_TO_ANTI_TARGETS:
                TARGET_TO_ANTI_TARGETS[str(t)] = list(args.anti_targets)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info("GPU selection: environment / SLURM; meta-config uses cuda:0.")

    multi_obj_root = os.path.join(os.environ["PROJECT_ROOT"], "genetic_gfn", "multi_objective")
    base_cfg_path = args.config_file
    if not os.path.isabs(base_cfg_path):
        base_cfg_path = os.path.join(multi_obj_root, base_cfg_path)
    with open(base_cfg_path, "r") as f:
        orig_config_dict = yaml.safe_load(f)
    orig_config_dict = orig_config_dict or {}

    hparam_path = None
    if args.hparam_config is not None:
        hparam_path = args.hparam_config
        if not os.path.isabs(hparam_path):
            hparam_path = os.path.join(multi_obj_root, args.hparam_config)
    config_dicts = prepare_hparam_config(orig_config_dict, hparam_path) if hparam_path else [orig_config_dict]

    
    all_cfg_paths = []
    gpu_ids = list(args.gpu_ids) if args.gpu_ids else [0]
    job_idx = 0

    if args.search_range is not None:
        config_dicts = config_dicts[args.search_range[0]:args.search_range[1]]
    for config_dict in config_dicts:
        # create log dirs
        model_name = os.path.basename(args.config_file).split(".")[0]
        log_dir = get_log_dir(
            method=f"genetic_gfn",
            model_name=model_name,
            exp_name="exp",
            suffix="-hparam" if args.hparam_config else ""
        )
        os.makedirs(log_dir, exist_ok=True)

        for target in targets:
            anti_targets = TARGET_TO_ANTI_TARGETS.get(str(target), [])
            anti_iter = anti_targets if anti_targets else [""]

            for anti_target in anti_iter:
                target_dir = os.path.join(log_dir, target)
                if anti_target:
                    target_dir = os.path.join(target_dir, f"anti-{anti_target}")
                os.makedirs(target_dir, exist_ok=True)

                for seed in args.seeds:
                    seed_log_dir = os.path.join(target_dir, f"seed-{seed}")
                    os.makedirs(seed_log_dir, exist_ok=True)

                    # 1) Write GeneticGFN hyperparam config_default YAML for this run
                    config_default_dict = dict(config_dict)
                    if args.oracle_url:
                        config_default_dict["oracle_url"] = args.oracle_url
                    config_default_path = os.path.join(seed_log_dir, "config_default.yaml")
                    with open(config_default_path, "w") as f:
                        yaml.safe_dump(config_default_dict, f, sort_keys=False)

                    # 2) Write a small runner meta-config consumed by this runner's run_hit()
                    run_name = f"{target}_hit_task_seed{seed}"
                    if anti_target:
                        run_name = f"{run_name}_anti-{anti_target}"
                    cfg_out = {
                        "seed": int(seed),
                        "target": str(target),
                        "anti_target": str(anti_target) if anti_target else "",
                        "objectives": f"{args.objectives_prefix},{target}",
                        "alpha_vector": str(args.alpha_vector),
                        "oracle_url": str(args.oracle_url) if args.oracle_url else "",
                        "max_oracle_calls": int(args.max_oracle_calls),
                        "freq_log": int(args.freq_log),
                        "output_dir": seed_log_dir,
                        "run_name": run_name,
                        "config_default_path": config_default_path,
                        # CPU parallelism for GA ops (joblib pool); passed through to run.py.
                        "n_jobs": int(args.n_jobs) if args.n_jobs is not None else -1,
                        "device": "cuda:0",
                        "gpu_id": int(gpu_ids[job_idx % len(gpu_ids)]),
                    }
                    job_idx += 1

                    cfg_file = os.path.join(seed_log_dir, "run_config.yaml")
                    with open(cfg_file, "w") as f:
                        yaml.safe_dump(cfg_out, f, sort_keys=False)
                    all_cfg_paths.append(cfg_file)

    total_workers = args.max_workers
    logging.info(f"Running {len(all_cfg_paths)} jobs with up to {total_workers} parallel workers")
    
    run_hits(cfg_paths=all_cfg_paths, max_workers=total_workers)
    logging.info(f"Completed {len(all_cfg_paths)} jobs")