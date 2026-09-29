import argparse
import os
import sys
from typing import List
import gc
import time
import logging

# Add parent directory to path for imports.
# genetic_chemalactica/ on path so `from utils.*` resolves to this package's utils/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from benchmark.benchmark_timer import BenchmarkTimer

from genetic_chemalactica.oracles.oracle import select_oracle
from benchmark.actives_loader import lead_seed_smiles
from benchmark.docking_oracle.docking import ANTITARGET_RECEPTORS
from benchmark.spec_tasks import spec_task_name

parp1_0 = lead_seed_smiles("parp1", 0)
parp1_1 = lead_seed_smiles("parp1", 1)
parp1_2 = lead_seed_smiles("parp1", 2)
fa7_0 = lead_seed_smiles("fa7", 0)
fa7_1 = lead_seed_smiles("fa7", 1)
fa7_2 = lead_seed_smiles("fa7", 2)
_5ht1b_0 = lead_seed_smiles("5ht1b", 0)
_5ht1b_1 = lead_seed_smiles("5ht1b", 1)
_5ht1b_2 = lead_seed_smiles("5ht1b", 2)
braf_0 = lead_seed_smiles("braf", 0)
braf_1 = lead_seed_smiles("braf", 1)
braf_2 = lead_seed_smiles("braf", 2)
jak2_0 = lead_seed_smiles("jak2", 0)
jak2_1 = lead_seed_smiles("jak2", 1)
jak2_2 = lead_seed_smiles("jak2", 2)

from genetic.genetic_utils import Entry, Pool, canonicalize
#from utils import generate_random_number

from logging_ import logger

from utils import set_seed
from utils.mol import find_valid_mols


def setup_logging(log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Remove all existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        # Ensure log file directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler with immediate flushing
        try:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.INFO)
            # Set formatter for file handler
            formatter = logging.Formatter(log_format, datefmt=date_format)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except Exception as e:
            # If file handler creation fails, log to stderr and continue with stdout only
            print(f"Warning: Could not create log file handler for {log_file}: {e}", file=sys.stderr)
    
    # Configure root logger - use force=True if available (Python 3.8+)
    try:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
            handlers=handlers,
            force=True  # Force reconfiguration even if already configured
        )
    except TypeError:
        # Python < 3.8 doesn't support force parameter
        # Manually configure the root logger
        root_logger.setLevel(logging.INFO)
        formatter = logging.Formatter(log_format, datefmt=date_format)
        for handler in handlers:
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
    
    # Return root logger to ensure all logging calls work
    logger = logging.getLogger()
    logger.info(f"Logging initialized. Log file: {log_file if log_file else 'stdout only'}")
    # Force flush to ensure message is written immediately
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
    return logger


class TempScheduler:

    def __init__(self, start_temp, end_temp, num_steps):
        assert start_temp >= end_temp
        self._start_temp = start_temp
        self._end_temp = end_temp
        self._temp_step = (start_temp - end_temp) / num_steps

    def step(self, i):
        self.temp = self._start_temp - self._temp_step * i
        self.temp = max(self.temp, self._end_temp + self._temp_step)


def props_dict_from_task(full_task_name: str):
    from genetic_chemalactica.utils.tasks import validate_task_name

    validate_task_name(full_task_name)
    prop_dict = {
        "hit.parp1": (
            ("QED", lambda: "[0.5, 0.94]"),
            ("SAS", lambda: "[1.0, 5.0]")
        ),
        "hit.fa7": (
            ("QED", lambda: "[0.5, 0.94]"),
            ("SAS", lambda: "[1.0, 5.0]")
        ),
        "hit.5ht1b": (
            ("QED", lambda: "[0.5, 0.94]"),
            ("SAS", lambda: "[1.0, 5.0]")
        ),
        "hit.braf": (
            ("QED", lambda: "[0.5, 0.94]"),
            ("SAS", lambda: "[1.0, 5.0]")
        ),
        "hit.jak2": (
            ("QED", lambda: "[0.5, 0.94]"),
            ("SAS", lambda: "[1.0, 5.0]")
        ),
        "lead.parp1_04_0": (
            ("SIMILAR", lambda: f"{parp1_0} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.parp1_04_1": (
            ("SIMILAR", lambda: f"{parp1_1} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.parp1_04_2": (
            ("SIMILAR", lambda: f"{parp1_2} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.parp1_06_0": (
            ("SIMILAR", lambda: f"{parp1_0} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.parp1_06_1": (
            ("SIMILAR", lambda: f"{parp1_1} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.parp1_06_2": (
            ("SIMILAR", lambda: f"{parp1_2} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.fa7_04_0": (
            ("SIMILAR", lambda: f"{fa7_0} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.fa7_04_1": (
            ("SIMILAR", lambda: f"{fa7_1} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.fa7_04_2": (
            ("SIMILAR", lambda: f"{fa7_2} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.fa7_06_0": (
            ("SIMILAR", lambda: f"{fa7_0} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.fa7_06_1": (
            ("SIMILAR", lambda: f"{fa7_1} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.fa7_06_2": (
            ("SIMILAR", lambda: f"{fa7_2} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.5ht1b_04_0": (
            ("SIMILAR", lambda: f"{_5ht1b_0} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.5ht1b_04_1": (
            ("SIMILAR", lambda: f"{_5ht1b_1} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.5ht1b_04_2": (
            ("SIMILAR", lambda: f"{_5ht1b_2} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.5ht1b_06_0": (
            ("SIMILAR", lambda: f"{_5ht1b_0} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.5ht1b_06_1": (
            ("SIMILAR", lambda: f"{_5ht1b_1} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.5ht1b_06_2": (
            ("SIMILAR", lambda: f"{_5ht1b_2} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.braf_04_0": (
            ("SIMILAR", lambda: f"{braf_0} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.braf_04_1": (
            ("SIMILAR", lambda: f"{braf_1} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.braf_04_2": (
            ("SIMILAR", lambda: f"{braf_2} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.braf_06_0": (
            ("SIMILAR", lambda: f"{braf_0} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.braf_06_1": (
            ("SIMILAR", lambda: f"{braf_1} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.braf_06_2": (
            ("SIMILAR", lambda: f"{braf_2} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.jak2_04_0": (
            ("SIMILAR", lambda: f"{jak2_0} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.jak2_04_1": (
            ("SIMILAR", lambda: f"{jak2_1} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.jak2_04_2": (
            ("SIMILAR", lambda: f"{jak2_2} [0.40,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.jak2_06_0": (
            ("SIMILAR", lambda: f"{jak2_0} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.jak2_06_1": (
            ("SIMILAR", lambda: f"{jak2_1} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ),
        "lead.jak2_06_2": (
            ("SIMILAR", lambda: f"{jak2_2} [0.60,1.00]"),
            ("QED", lambda: "[0.60,1.00]"),
            ("SAS", lambda: "[1.00,4.00]")
        ), 
        **{
            spec_task_name(at): (
                ("QED", lambda: "[0.40,1.00]"),
                ("SAS", lambda: "[1.00,4.00]"),
                ("DOCKING SCORE", lambda: "[10.67,20.00]"),
            )
            for at in ANTITARGET_RECEPTORS
        },
    }
    prop_list = prop_dict[full_task_name]

    task_prompt = ""
    for prop in prop_list:
        start_tag = f"[{prop[0]}]"
        end_tag = f"[/{prop[0]}]"
        value = prop[1]()
        task_prompt += f"{start_tag}{value}{end_tag}"

    return task_prompt


def create_prompts(
    pool: Pool,
    model_args,
    genetic_args,
    include_start_token: bool,
    task_name: str
):
    prompts = []
    for n in range(genetic_args.num_prompts * 2):
        similar_entries = pool.random_subset(genetic_args.num_similars)
        prompt = ""

        # add similar molecules in the prompt
        for entry in similar_entries:
            sim_value = np.random.uniform(genetic_args.sim_vals[0], genetic_args.sim_vals[1])
            prompt += f"{model_args.sim_tags[0]}{entry.mol} {sim_value:.2f}{model_args.sim_tags[1]}"

        prompt += props_dict_from_task(task_name)

        # add start of molecule tag
        if include_start_token:
            prompt += model_args.mol_tags[0]
        prompts.append(prompt)
    return prompts


def resolve_oracle_task_name(oracle_cfg) -> str:
    task_name = OmegaConf.select(oracle_cfg, "task_name", default=None)
    if task_name:
        return str(task_name)
    legacy = OmegaConf.select(oracle_cfg, "name", default=None)
    if legacy:
        return str(legacy)
    raise ValueError("oracle.task_name is required in config")


def search(args):
    log_dir = args.oracle.log_dir
    task_name = resolve_oracle_task_name(args.oracle)
    # Setup logging
    run_log_path = os.path.join(log_dir, "run.log")
    setup_logging(run_log_path)

    seed = OmegaConf.select(args, "seed", default=None)
    if seed is None:
        raise ValueError("Config must set `seed` (genetic_runner writes it per run).")
    seed = int(seed)
    set_seed(seed)
    logger.info(f"Running Genetic with seed={seed}")
    timer = BenchmarkTimer(log_dir=log_dir)

    checkpoint_path = os.environ.get("HF_LOCAL_MODEL_DIR", "").strip() or str(args.model.checkpoint_path)
    tokenizer_path = os.environ.get("HF_LOCAL_TOKENIZER_DIR", "").strip() or str(args.model.tokenizer_path)

    # Load model with memory-efficient settings
    # Use device_map dict format for explicit single-device placement
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": args.device},  # Explicitly map all layers to the specified device
        attn_implementation="sdpa",  # Use efficient SDPA attention
        local_files_only=bool(os.environ.get("HF_LOCAL_MODEL_DIR", "").strip()),
    )
    model.eval()  # Set to eval mode to disable gradients and reduce memory
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_path,
        padding_side="left",
        local_files_only=bool(os.environ.get("HF_LOCAL_TOKENIZER_DIR", "").strip()),
    )
    pad_token_id = tokenizer.convert_tokens_to_ids(str(args.model.pad_token))
    if pad_token_id != tokenizer.unk_token_id:
        tokenizer.pad_token_id = pad_token_id

    # Optional: HTTP DockingVina service URL. If absent, property_computers uses local quickvina.
    vina_url = OmegaConf.select(args, "vina_url", default=None)
    use_oracles_app = OmegaConf.select(args, "oracle.use_oracles_app", default=False)
    resume = bool(OmegaConf.select(args.oracle, "resume", default=False))
    oracle = select_oracle(
        reward_type=args.oracle.reward_type,
        task_name=task_name,
        log_dir=args.oracle.log_dir,
        max_oracle_calls=args.oracle.max_calls,
        vina_url=vina_url,
        use_oracles_app=use_oracles_app,
        bench_timer=timer,
        seed=seed,
        resume=resume,
    )
    pool = Pool(args.genetic.pool_size)
    if oracle.mol_buffer:
        pool_entries = []
        skipped = 0
        for smiles, props in oracle.mol_buffer.items():
            try:
                prop_score = [props.get(name) for name in oracle.computer_names]
                score = oracle.score_computer(prop_score)
                pool_entries.append(Entry(smiles=smiles, score=score))
            except Exception:
                skipped += 1
        if pool_entries:
            pool.add(pool_entries)
        logger.info(
            "Resumed %s unique mols from %s (remaining budget %s/%s), pool=%s, skipped_unscorable=%s",
            len(oracle.mol_buffer),
            oracle.log_file_path,
            oracle.max_oracle_calls - len(oracle.mol_buffer),
            oracle.max_oracle_calls,
            len(pool),
            skipped,
        )
    
    # Track oracle call statistics
    total_oracle_calls = 0
    total_wasted_calls = 0

    entries: List[Entry] = []

    num_iter = 0
    oracle_calls_trend = []
    while not oracle.finish():
        oracle_calls_trend.append(len(oracle.mol_buffer))
        if len(oracle_calls_trend) > 10:
            if oracle_calls_trend[-1] == oracle_calls_trend[-10]:
                logger.info("Stopping: no new molecules generated")
                break
        num_iter = len(oracle.mol_buffer) // args.genetic.num_prompts
        # temp_scheduler.step(num_iter)
        logger.info(f"Iter: {num_iter}")

        # Generation timing: prompts/setup vs LM forward vs post-process (add gen_train phase here if you fine-tune).
        with timer.phase("gen_prompts"):
            prompts = create_prompts(
                pool,
                args.model,
                args.genetic,
                include_start_token=True,
                task_name=task_name
            )

            gen_config = OmegaConf.to_container(args.generation)
            gen_batch_size = getattr(args.model, 'gen_batch_size', 32)
            if gen_batch_size is None or gen_batch_size > len(prompts):
                gen_batch_size = len(prompts)

        with timer.phase("gen_model"):
            all_generated_tokens = []
            for start_idx in range(0, len(prompts), gen_batch_size):
                end_idx = min(start_idx + gen_batch_size, len(prompts))
                batch_prompts = prompts[start_idx:end_idx]

                data = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(args.device)

                generated_texts = model.generate(
                    **data,
                    **gen_config
                )
                batch_generated_tokens = tokenizer.batch_decode(generated_texts)
                all_generated_tokens.extend(batch_generated_tokens)

                del data
                del generated_texts
                torch.cuda.empty_cache()

        with timer.phase("gen_post"):
            generated_tokens = all_generated_tokens
            raw_mols = find_valid_mols(generated_tokens, args.model.mol_tags[0], args.model.mol_tags[1])
            paired_mols = []
            paired_prompts = []
            n_invalid = 0
            for mol, prompt in zip(raw_mols, prompts):
                canon = canonicalize(mol)
                if canon:
                    paired_mols.append(canon)
                    paired_prompts.append(prompt)
                else:
                    n_invalid += 1
            if n_invalid:
                logger.info(f"Skipped {n_invalid} invalid/empty generations this round")

        if not paired_mols:
            logger.info(f"Iter {num_iter}: no valid molecules generated, retrying")
            continue

        # keep only unique molecules
        unique_mols, unique_inds = np.unique(paired_mols, return_index=True)
        unique_prompts = np.array(paired_prompts)[unique_inds]
        assert len(unique_prompts) == len(unique_mols)

        # keep only the unique molecules and prompts
        remaining_num_mols = args.genetic.num_prompts - len(entries)
        remaining_budget = args.oracle.max_calls - len(oracle.mol_buffer)
        if remaining_budget <= 0:
            break
        batch_limit = min(remaining_num_mols, remaining_budget)
        unique_mols = unique_mols[:batch_limit]
        unique_prompts = unique_prompts[:batch_limit]

        # Count wasted calls (molecules already in buffer)
        wasted_calls = sum(1 for mol in unique_mols if mol in oracle.mol_buffer)
        new_mols_count = len(unique_mols) - wasted_calls
        
        # Oracle timing is split inside the oracle: oracle_props (new molecules only) vs oracle_scoring
        # (reward + log for the whole batch, including duplicates / “wasted” calls).
        oracle_call_start = time.time()
        scores = oracle(unique_mols, unique_prompts)
        oracle_call_time = time.time() - oracle_call_start
        
        # Update statistics
        oracle_calls_after = len(oracle.mol_buffer)
        total_oracle_calls += len(unique_mols)
        total_wasted_calls += wasted_calls
        
        for mol, score in zip(unique_mols, scores, strict=True):
            try:
                entries.append(Entry(smiles=mol, score=score))
            except Exception:
                logger.warning("Skipping unscorable mol after oracle: %r", mol)
                continue

        # Counts / oracle stats go to the run log only; timing_profile.csv stays time-only (phase seconds).
        logger.info(
            f"Iter {num_iter}: oracle_time={oracle_call_time:.2f}s, unique_mols={len(unique_mols)}, "
            f"new_mols={new_mols_count}, wasted={wasted_calls}, total_wasted={total_wasted_calls}, "
            f"cumulative_eval_time={oracle.time_spent_on_evaluation:.2f}s, buffer={oracle_calls_after}/{args.oracle.max_calls}"
        )
        timer.log_iteration(num_iter)

        # add the molecules to the pool
        if len(entries) == args.genetic.num_prompts:
            num_iter += 1
            pool.add(entries)
            entries.clear()
    
    # Final summary
    final_oracle_calls = len(oracle.mol_buffer)
    wasted_percentage = (total_wasted_calls/total_oracle_calls*100) if total_oracle_calls > 0 else 0
    
    logger.info(f"Final: total_wasted={total_wasted_calls} ({wasted_percentage:.1f}%), total_oracle_time={oracle.time_spent_on_evaluation:.2f}s, calls={final_oracle_calls}/{args.oracle.max_calls}")
    timer.log_summary()
    
    logger.info(f"Run log saved to: {run_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
       
    parser.add_argument("--config_file", required=False, type=str)
    parser.add_argument("--seeds", nargs="+", required=False, type=int)
    parser.add_argument("--oracle.task_name", required=False, type=str)
    #parser.add_argument("--oracle_calls", required=False, type=int)
    parser.add_argument("--n_gpus", required=False, default=1, type=int)
    parser.add_argument("--hparam_config", type=str, required=False, default=None)

    cmd_args = parser.parse_args()

    args = OmegaConf.load(cmd_args.config_file)
    # Set DOCKING_VINA_URL from config (vina_url written by genetic_runner) so docking code sees it
    vina_url = OmegaConf.select(args, "vina_url", default=None)
    if vina_url:
        os.environ["DOCKING_VINA_URL"] = str(vina_url)
    search(args)
    
    