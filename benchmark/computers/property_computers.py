from __future__ import annotations

import logging
import os
from functools import partial

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, DataStructs, rdMolDescriptors
from rdkit.Chem.QED import qed

from benchmark.docking_oracle.docking_vina_client import DockingOracleClient
from benchmark.computers.simulated_properties import quickvina_predictor
from benchmark.synthesizability import sascorer
RDLogger.DisableLog("rdApp.*")


# Public aliases (simple mechanism for algorithms)
def QED(rdkit_mols, verb: bool = True):
    return compute_qed(rdkit_mols, verb=verb)


def SA(rdkit_mols, verb: bool = True):
    return compute_sas(rdkit_mols, verb=verb)


def SIMILARITY(rdkit_mols, rdkit_mol1, verb: bool = True):
    return compute_similarity(rdkit_mols, rdkit_mol1=rdkit_mol1, verb=verb)


def select_prop_computer(computer_name: str, vina_url: str | None = None, seed: int = 0):
    name_to_computer = {
        "QED": compute_qed,
        "TPSA": compute_tpsa,
        "SAS": compute_sas,
        "CLOGP": compute_clogp,
        "WEIGHT": compute_weight,
        "FORMULA": compute_formula,
        "NUMAROMATICRINGS": compute_num_aromatic_rings,
        "RINGCOUNT": compute_num_rings,
    }
    if computer_name in name_to_computer:
        return name_to_computer[computer_name]

    arg = ".".join(computer_name.split(".")[1:])
    base = computer_name.split(".")[0]
    if base == "SIMILAR":
        mol = Chem.MolFromSmiles(arg)
        return partial(compute_similarity, rdkit_mol1=mol)
    if base == "DOCKING":
        target = arg
        return partial(compute_quickvina_docking_score, target=target, vina_url=vina_url, seed=seed)

    raise ValueError(f"Oracle with name {computer_name} does not exist.")


def compute_qed_sas_docking(rdkit_mols, target: str, vina_url: str | None = None, seed: int = 0):
    computer_names = ["QED", "SAS", f"DOCKING.{target}"]
    return dynamic_computer(rdkit_mols, computer_names, vina_url=vina_url, seed=seed)


def geam_docking_oracle(rdkit_mols, target: str, verb: bool = True, vina_url: str | None = None, seed: int = 0):
    docking_scores = compute_quickvina_docking_score(
        rdkit_mols, target, verb=verb, vina_url=vina_url, seed=seed
    )
    qed_scores = compute_qed(rdkit_mols, verb=verb)
    sa_scores = compute_sas(rdkit_mols)
    trans_sa_scores = (10 - sa_scores) / 9
    aggregated_scores = (np.clip(docking_scores, 0, 20) / 20) * qed_scores * trans_sa_scores
    return aggregated_scores, docking_scores, qed_scores, sa_scores


def compute_quickvina_docking_score(
    rdkit_mols,
    target: str,
    verb: bool = True,
    vina_url: str | None = None,
    seed: int = 0,
):
    valid_indices = []
    valid_mols = []
    for i, rdkit_mol in enumerate(rdkit_mols):
        if rdkit_mol is not None:
            valid_indices.append(i)
            valid_mols.append(rdkit_mol)

    scores = np.zeros(len(rdkit_mols), dtype=np.float32)
    if len(valid_mols) == 0:
        return scores

    smiles_list = [Chem.MolToSmiles(rdkit_mol) for rdkit_mol in valid_mols]

    docking_vina_url = vina_url or os.environ.get("DOCKING_VINA_URL")
    logging.getLogger(__name__).info(
        "docking_vina_url: %s (from %s)",
        docking_vina_url,
        "request param" if vina_url else "env DOCKING_VINA_URL",
    )
    if docking_vina_url:
        client = DockingOracleClient(docking_vina_url, target)
        valid_scores = client.predict(smiles_list, seed=seed)
        valid_scores = -np.array(valid_scores)
        valid_scores = np.clip(valid_scores, 0, None)
        scores[valid_indices] = valid_scores
    else:
        predictor = quickvina_predictor(target)
        valid_scores = -np.array(predictor.predict(smiles_list, seed=seed))
        valid_scores = np.clip(valid_scores, 0, None)
        scores[valid_indices] = valid_scores
    return scores


def dynamic_computer(
    rdkit_mols,
    computer_names,
    verb: bool = True,
    vina_url: str | None = None,
    seed: int = 0,
):
    scores_dict = {}
    for computer_n in computer_names:
        prop_computer = select_prop_computer(computer_n, vina_url=vina_url, seed=seed)
        p_scores = prop_computer(rdkit_mols, verb=verb)
        scores_dict[computer_n] = p_scores
    return scores_dict


def compute_qed(rdkit_mols, verb: bool = True):
    qed_scores = []
    for rdkit_mol in rdkit_mols:
        try:
            qed_scores.append(qed(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute QED for", rdkit_mol, e)
            qed_scores.append(None)
    return np.array(qed_scores)


def compute_clogp(rdkit_mols, verb: bool = True):
    logp_scores = []
    for rdkit_mol in rdkit_mols:
        try:
            logp_scores.append(Crippen.MolLogP(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute CLOGP for", rdkit_mol, e)
            logp_scores.append(None)
    return np.array(logp_scores)


def compute_tpsa(rdkit_mols, verb: bool = True):
    tpsa_scores = []
    for rdkit_mol in rdkit_mols:
        try:
            tpsa_scores.append(rdMolDescriptors.CalcTPSA(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute TPSA for", rdkit_mol, e)
            tpsa_scores.append(None)
    return np.array(tpsa_scores)


def compute_weight(rdkit_mols, verb: bool = True):
    weights = []
    for rdkit_mol in rdkit_mols:
        try:
            weights.append(rdMolDescriptors.CalcExactMolWt(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute weight for", rdkit_mol, e)
            weights.append(None)
    return np.array(weights)


def compute_num_aromatic_rings(rdkit_mols, verb: bool = True):
    num_arom_rings = []
    for rdkit_mol in rdkit_mols:
        try:
            num_arom_rings.append(rdMolDescriptors.CalcNumAromaticRings(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute aromatic rings for", rdkit_mol, e)
            num_arom_rings.append(None)
    return np.array(num_arom_rings)


def compute_num_rings(rdkit_mols, verb: bool = True):
    num_rings = []
    for rdkit_mol in rdkit_mols:
        try:
            num_rings.append(rdMolDescriptors.CalcNumRings(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute ringcount for", rdkit_mol, e)
            num_rings.append(None)
    return np.array(num_rings)


def compute_formula(rdkit_mols, verb: bool = True):
    formulas = []
    for rdkit_mol in rdkit_mols:
        try:
            formulas.append(rdMolDescriptors.CalcMolFormula(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute formula for", rdkit_mol, e)
            formulas.append(None)
    return np.array(formulas)


def compute_sas(rdkit_mols, verb: bool = True):
    sa_scores = []
    for rdkit_mol in rdkit_mols:
        try:
            sa_scores.append(sascorer.calculateScore(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute SA score for", rdkit_mol, e)
            sa_scores.append(None)
    return np.array(sa_scores)


def compute_fing(rdkit_mols, verb: bool = True):
    fings = []
    for rdkit_mol in rdkit_mols:
        if rdkit_mol is None:
            fings.append(None)
            continue
        try:
            fings.append(AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, 2, nBits=2048))
        except Exception as e:
            if verb:
                print("Could not compute fingerprint for", rdkit_mol, e)
            fings.append(None)
    return fings


def _tanimoto_or_none(fp, ref_fp):
    if fp is None or ref_fp is None:
        return None
    try:
        return DataStructs.TanimotoSimilarity(fp, ref_fp)
    except Exception:
        return None


def compute_similarity(rdkit_mols, rdkit_mol1, verb: bool = True):
    if verb:
        print(f"Computing similarity between {len(rdkit_mols)} generated molecules and {rdkit_mol1} seed molecule.")
    fings = compute_fing(rdkit_mols)
    fing1 = compute_fing([rdkit_mol1])[0]
    return np.array([_tanimoto_or_none(f, fing1) for f in fings])


def compute_similarity_fing(fings, fing1):
    return np.array([_tanimoto_or_none(f, fing1) for f in fings])

