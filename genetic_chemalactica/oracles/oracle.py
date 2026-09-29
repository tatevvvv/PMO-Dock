from contextlib import nullcontext
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import yaml
import time

import numpy as np
import pandas as pd
import requests
from rdkit import Chem

from logging_ import logger

import os
from functools import partial
import math

import torch

from benchmark.computers.property_computers import dynamic_computer
from genetic_chemalactica.utils.tasks import (
    task_name2computer_names,
    task_name2hit_ranges,
    select_sigma,
    validate_task_name,
    validate_reward_type,
)
from benchmark.rewards import hit_reward, compute_geam_reward
from genetic_chemalactica.utils.rewards import hit_docking_score_reward, hit_spec_reward


def _parse_csv_number(val):
    if val is None:
        return None
    if isinstance(val, (float, np.floating)) and np.isnan(val):
        return None
    text = str(val).strip()
    if text == "" or text.lower() in {"none", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_mol_buffer_from_csv(log_file_path: str, computer_names: List[str], sep: str = ";") -> Dict:
    """First-seen unique mols from an oracle CSV, with property dicts (no re-docking)."""
    df = pd.read_csv(log_file_path, sep=sep, dtype=str, on_bad_lines="skip")
    if df.empty or "mol" not in df.columns:
        return {}

    col_for = {}
    for name in computer_names:
        if name in df.columns:
            col_for[name] = name
            continue
        if name.startswith("SIMILAR"):
            matches = [c for c in df.columns if str(c).startswith("SIMILAR")]
            if len(matches) == 1:
                col_for[name] = matches[0]
    if len(col_for) != len(computer_names):
        missing = [n for n in computer_names if n not in col_for]
        raise ValueError(f"CSV {log_file_path} missing property columns {missing}; have {list(df.columns)}")

    buffer: Dict = {}
    for _, row in df.iterrows():
        mol_raw = row.get("mol")
        if mol_raw is None or (isinstance(mol_raw, float) and np.isnan(mol_raw)):
            continue
        mol = str(mol_raw).strip()
        if not mol or mol.lower() == "nan" or mol in buffer:
            continue
        buffer[mol] = {name: _parse_csv_number(row.get(col_for[name])) for name in computer_names}
    return buffer


def select_oracle(
    task_name: str,
    log_dir: str,
    reward_type: str="hit",
    max_oracle_calls: int=1000,
    freq_log: int=100,
    vina_url: str | None = None,
    use_oracles_app: bool = False,
    bench_timer: Optional[Any] = None,
    seed: int = 0,
    resume: bool = False,
):
    validate_task_name(task_name)
    reward_type = validate_reward_type(task_name, reward_type)
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    log_file_path = os.path.join(
        log_dir, f'{rank}_mols.csv'
    )
    computer_names = task_name2computer_names(task_name)
    sigmas = [select_sigma(comp) for comp in computer_names]
    hit_ranges = task_name2hit_ranges(task_name)

    # Allow disabling the internal Flask oracle service and running locally in-process.
    # Env var is a convenient override for SLURM jobs / scripts.
    disable_oracles_app_env = os.environ.get("ORACLES_APP_DISABLE", "").strip().lower() in {"1", "true", "yes", "y"}
    use_oracles_app = bool(use_oracles_app) and not disable_oracles_app_env

    url = None
    if use_oracles_app:
        oracles_host = os.environ.get("ORACLES_APP_HOST", "127.0.0.1")
        oracles_port = int(os.environ.get("ORACLES_APP_PORT", "5454"))
        url = f"{oracles_host}:{oracles_port}/dynamic"
        print(f"Oracle URL: {url}")
    else:
        print("Oracle mode: local (no ORACLES_APP HTTP service)")

    if task_name.startswith("hit."):
        if reward_type == "hit":
            score_computer = partial(hit_reward, sigmas=sigmas, hit_ranges=hit_ranges)
        elif reward_type == "max":
            score_computer = partial(hit_docking_score_reward, sigmas=sigmas, hit_ranges=hit_ranges)
        elif reward_type == "geam":
            score_computer = compute_geam_reward
        else:
            raise ValueError(f"Invalid reward type {reward_type}")
    elif task_name.startswith("spec."):
        score_computer = partial(hit_spec_reward, sigmas=sigmas, hit_ranges=hit_ranges)
    elif task_name.startswith("lead."):
        score_computer = partial(hit_docking_score_reward, sigmas=sigmas, hit_ranges=hit_ranges)
    else:
        raise ValueError(f"Oracle name {task_name} does not exist")

    oracle_cls = DynamicOracle if use_oracles_app else LocalDynamicOracle
    return oracle_cls(
        computer_names=computer_names,
        log_file_path=log_file_path,
        url=url,
        max_oracle_calls=max_oracle_calls,
        freq_log=freq_log,
        score_computer=score_computer,
        vina_url=vina_url,
        bench_timer=bench_timer,
        seed=seed,
        resume=resume,
    )


class DynamicOracle:

    def __init__(
        self,
        computer_names: List[str],
        log_file_path: str,
        url: str | None,
        score_computer,
        max_oracle_calls: int,
        freq_log: int,
        additional_args: dict={},
        vina_url: str | None = None,
        bench_timer: Optional[Any] = None,
        seed: int = 0,
        resume: bool = False,
    ):
        self.computer_names = computer_names
        self.log_file_path = log_file_path
        self.url = url
        self.score_computer = score_computer
        self.max_oracle_calls = max_oracle_calls
        self.freq_log = freq_log
        self.additional_args = additional_args
        self._bench_timer = bench_timer
        self.seed = int(seed)
        if vina_url is not None:
            self.additional_args["vina_url"] = vina_url
        self.additional_args["seed"] = self.seed
        self.sep = ";"
        self.mol_buffer: Dict = {}
        self.time_spent_on_evaluation = 0.0
        self.vina_url = vina_url
        heading = f"mol{self.sep}prompt{self.sep}score{self.sep}{self.sep.join(self.computer_names)}\n"
        can_resume = (
            resume
            and os.path.isfile(self.log_file_path)
            and os.path.getsize(self.log_file_path) > 0
        )
        if can_resume:
            self.mol_buffer = load_mol_buffer_from_csv(self.log_file_path, self.computer_names, sep=self.sep)
            self.out_file = open(self.log_file_path, "a")
        else:
            self.out_file = open(self.log_file_path, "w")
            self.out_file.write(heading)
            self.out_file.flush()

    def _bench_phase(self, name: str):
        t = self._bench_timer
        if t is None:
            return nullcontext()
        return t.phase(name)

    def __call__(
        self,
        mols: List[str],
        prompts: List[str]
    ) -> List[float]:
        start_time = time.time()
        res = self.evaluate(mols, prompts)
        time_spent = time.time() - start_time
        self.time_spent_on_evaluation += time_spent
        logger.info(f"Total time on oracle eval {self.time_spent_on_evaluation / 1000:.4f}, cur time: {time_spent / 1000:.4f}")
        return res

    def evaluate(
        self,
        mols: List[str],
        prompts: List[str]
    ) -> List[float]:
        """Calculate the oracle value for mols

        Args:
            mols (List[str]): the list of mols to calculate oracle value for
            prompts (List[str]): the prompts used for generating mols

        Returns:
            List[float]: the list of oracle value calculated
        """
        new_mols = []
        for mol in mols:
            if mol not in self.mol_buffer:
                new_mols.append(mol)

        data = {"mols": new_mols, "computer_names": self.computer_names}
        data.update(self.additional_args)

        with self._bench_phase("oracle_props"):
            if len(new_mols) > 0:
                if not self.url:
                    raise ValueError("DynamicOracle requires url when using ORACLES_APP mode.")
                while True:
                    try:
                        res = requests.post(f"http://{self.url}", json=data)
                        if res.status_code == 200:  # Check if the response is successful
                            break  # Exit the loop on success
                        else:
                            logger.info(f"Server returned status code {res.status_code}, retrying...")
                            time.sleep(5)
                    except Exception as e:
                        logger.info(f"Request failed: {e}, retrying...")
                        time.sleep(5)

                response = res.json()
                for i, mol in enumerate(new_mols):
                    self.mol_buffer[mol] = {}
                    for prop_name, values in response.items():
                        self.mol_buffer[mol][prop_name] = values[i]

        # Reward + CSV for every molecule in the batch (includes duplicates already in buffer).
        with self._bench_phase("oracle_scoring"):
            errors = []
            rank_output_rows = []
            for prompt, mol in zip(prompts, mols, strict=True):
                prop_score = [self.mol_buffer[mol][comp_name] for comp_name in self.computer_names] # keep the order of the computers
                err = self.score_computer(prop_score)
                errors.append(err)

                out_row = f"{mol}{self.sep}{prompt}{self.sep}{err}{self.sep}{self.sep.join(map(str, prop_score))}\n"
                rank_output_rows.append(out_row)

            self.out_file.write("".join(rank_output_rows))
            self.out_file.flush()

        return errors

    def finish(self, *args, **kwargs) -> bool:
        """Returns whether or not the oracle finished evaluating mols 

        Returns:
            bool: Whether or not the oracle finished evaluating mols
        """
        return len(self.mol_buffer) >= self.max_oracle_calls


class LocalDynamicOracle(DynamicOracle):
    """
    In-process oracle: computes requested properties locally (no HTTP service).
    Keeps the same external interface as DynamicOracle so the genetic loop is unchanged.
    """

    def evaluate(self, mols: List[str], prompts: List[str]) -> List[float]:
        new_mols = [mol for mol in mols if mol not in self.mol_buffer]

        with self._bench_phase("oracle_props"):
            if len(new_mols) > 0:
                rdkit_mols = [Chem.MolFromSmiles(s) for s in new_mols]
                scores_dict = dynamic_computer(
                    rdkit_mols,
                    self.computer_names,
                    vina_url=self.additional_args.get("vina_url"),
                    seed=self.additional_args.get("seed", self.seed),
                )
                for i, mol in enumerate(new_mols):
                    self.mol_buffer[mol] = {prop_name: values[i] for prop_name, values in scores_dict.items()}

        with self._bench_phase("oracle_scoring"):
            errors = []
            rank_output_rows = []
            for prompt, mol in zip(prompts, mols, strict=True):
                prop_score = [self.mol_buffer[mol][comp_name] for comp_name in self.computer_names]
                err = self.score_computer(prop_score)
                errors.append(err)

                out_row = f"{mol}{self.sep}{prompt}{self.sep}{err}{self.sep}{self.sep.join(map(str, prop_score))}\n"
                rank_output_rows.append(out_row)

            self.out_file.write("".join(rank_output_rows))
            self.out_file.flush()
        return errors