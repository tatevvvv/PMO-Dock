"""
DEPRECATED: algorithms should import benchmark computers from `benchmark.computers`.

This module remains as a thin compatibility shim.
"""

import json
import os
import subprocess

from benchmark.computers.property_computers import *  # noqa: F403
from benchmark.docking_oracle.docking_vina_client import DockingOracleClient
from benchmark.paths import resolve_from_project_root


def select_prop_computer(computer_name, vina_url=None, seed=0):
    name_to_computer = {
        "QED": compute_qed,
        "TPSA": compute_tpsa,
        "SAS": compute_sas,
        "CLOGP": compute_clogp,
        "WEIGHT": compute_weight,
        "FORMULA": compute_formula,
        "NUMAROMATICRINGS": compute_num_aromatic_rings,
        "RINGCOUNT": compute_num_rings,

        # Binding predictors
        "JNK3": partial(predict_binding_score, protein="jnk3"),
        "DRD2": partial(predict_binding_score, protein="drd2"),
        "GSK3B": partial(predict_binding_score, protein="gsk3b"),
    }
    if computer_name in name_to_computer:
        return name_to_computer[computer_name]
    
    arg = ".".join(computer_name.split(".")[1:])
    computer_name = computer_name.split(".")[0]
    if computer_name == "SIMILAR":
        mol = Chem.MolFromSmiles(arg)
        return partial(compute_similarity, rdkit_mol1=mol)
    elif computer_name == "DOCKING":
        target = arg
        return partial(compute_quickvina_docking_score, target=target, vina_url=vina_url, seed=seed)
    
    raise ValueError(f"Oracle with name {computer_name} does not exist.")


def rel_err(measured, real):
    return np.abs(measured - real) / real


def compute_qed_sas_docking(rdkit_mols, target: str, vina_url=None, seed=0):
    computer_names = ["QED", "SAS", f"DOCKING.{target}"]
    return dynamic_computer(rdkit_mols, computer_names, vina_url=vina_url, seed=seed)


def geam_docking_oracle(rdkit_mols, target: str, verb: bool=True, vina_url=None, seed=0):
    docking_scores = compute_quickvina_docking_score(
        rdkit_mols, target, verb=verb, vina_url=vina_url, seed=seed
    )
    qed_scores = compute_qed(rdkit_mols, verb=verb)
    sa_scores = compute_sas(rdkit_mols)

    # Formula used in GEAM paper
    trans_sa_scores = (10 - sa_scores) / 9
    aggregated_scores = (np.clip(docking_scores, 0, 20) / 20) * qed_scores * trans_sa_scores
    return aggregated_scores, docking_scores, qed_scores, sa_scores

# def compute_quickvina_docking_score(rdkit_mols, target: str):
#     predictor = quickvina_predictor(target)
#     smiles_list = [Chem.MolToSmiles(rdkit_mol) for rdkit_mol in rdkit_mols]
#     scores = -np.array(predictor.predict(smiles_list))
#     return np.clip(scores, 0, None)
def compute_quickvina_docking_score(rdkit_mols, target: str, verb=True, vina_url=None, seed=0):
    """
    Compute docking scores using either DockingVina service or local quickvina predictor.
    Skips invalid molecules (None) to save oracle budget.
    
    Args:
        rdkit_mols: List of RDKit molecule objects (may contain None for invalid molecules)
        target: Target protein name (e.g., 'parp1', 'fa7', '5ht1b', 'braf', 'jak2')
        verb: Verbose flag (default: True)
        vina_url: Optional docking service URL (takes precedence over env DOCKING_VINA_URL)
        seed: Docking RNG seed (experiment seed)
    
    Returns:
        Array of docking scores (clipped to >= 0). Invalid molecules get score 0.0.
    """
    # Filter out None values and track valid indices
    valid_indices = []
    valid_mols = []
    for i, rdkit_mol in enumerate(rdkit_mols):
        if rdkit_mol is not None:
            valid_indices.append(i)
            valid_mols.append(rdkit_mol)
    
    # Initialize result array with default score (0.0) for all molecules
    scores = np.zeros(len(rdkit_mols), dtype=np.float32)
    
    # If no valid molecules, return zeros
    if len(valid_mols) == 0:
        return scores
    
    # Convert valid molecules to SMILES
    smiles_list = [Chem.MolToSmiles(rdkit_mol) for rdkit_mol in valid_mols]
    
    docking_vina_url = vina_url or os.environ.get("DOCKING_VINA_URL")
    logging.getLogger(__name__).info(
        "docking_vina_url: %s (from %s)",
        docking_vina_url,
        "request param" if vina_url else "env DOCKING_VINA_URL",
    )
    if docking_vina_url:
        # Use docking HTTP service (same client as benchmark) for valid molecules only
        client = DockingOracleClient(docking_vina_url, target)
        valid_scores = client.predict(smiles_list, seed=seed)
        valid_scores = -np.array(valid_scores)  # Negative affinity to positive score
        valid_scores = np.clip(valid_scores, 0, None)
        scores[valid_indices] = valid_scores
    else:
        # Use local quickvina predictor (original behavior, same as benchmark)
        predictor = quickvina_predictor(target)
        valid_scores = -np.array(predictor.predict(smiles_list, seed=seed))
        valid_scores = np.clip(valid_scores, 0, None)
        scores[valid_indices] = valid_scores
    
    return scores


def dynamic_computer(rdkit_mols, computer_names, verb=True, vina_url=None, seed=0):
    scores_dict = {}
    for computer_n in computer_names:
        prop_computer = select_prop_computer(computer_n, vina_url=vina_url, seed=seed)
        p_scores = prop_computer(rdkit_mols, verb=verb)
        scores_dict[computer_n] = p_scores
    return scores_dict


def compute_qed(rdkit_mols, verb=True):
    qed_scores = []
    for rdkit_mol in rdkit_mols:
        try:
            qed_scores.append(qed(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute QED for", rdkit_mol, e)
            qed_scores.append(None)
    return np.array(qed_scores)


def compute_clogp(rdkit_mols, verb=True):
    logp_scores = []
    for rdkit_mol in rdkit_mols:
        try:
            logp_scores.append(Crippen.MolLogP(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute CLOGP for", rdkit_mol, e)
            logp_scores.append(None)
    return np.array(logp_scores)


def compute_tpsa(rdkit_mols, verb=True):
    tpsa = []
    for rdkit_mol in rdkit_mols:
        try:
            tpsa.append(rdMolDescriptors.CalcTPSA(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute TPSA for", rdkit_mol, e)
            tpsa.append(None)
    return np.array(tpsa)
        
        
def compute_weight(rdkit_mols, verb=True):
    weights = []
    for rdkit_mol in rdkit_mols:
        try:
            weights.append(rdMolDescriptors.CalcExactMolWt(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute weight for", rdkit_mol, e)
            weights.append(None)
    return np.array(weights)


def compute_num_aromatic_rings(rdkit_mols, verb=True):
    num_arom_rings = []
    for rdkit_mol in rdkit_mols:
        try:
            num_arom_rings.append(rdMolDescriptors.CalcNumAromaticRings(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute weight for", rdkit_mol, e)
            num_arom_rings.append(None)
    return np.array(num_arom_rings)
    # return np.vectorize(rdMolDescriptors.CalcNumAromaticRings)(rdkit_mols)


def compute_num_rings(rdkit_mols, verb=True):
    num_rings = []
    for rdkit_mol in rdkit_mols:
        try:
            num_rings.append(rdMolDescriptors.CalcNumRings(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute weight for", rdkit_mol, e)
            num_rings.append(None)
    return np.array(num_rings)
    # return np.vectorize(rdMolDescriptors.CalcNumRings)(rdkit_mols)


def compute_formula(rdkit_mols, verb=True):
    formulas = []
    for rdkit_mol in rdkit_mols:
        try:
            formulas.append(rdMolDescriptors.CalcMolFormula(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute formula for", rdkit_mol, e)
            formulas.append(None)
    return np.array(formulas)


def compute_sas(rdkit_mols, verb=True):
    sa_scores = []
    for rdkit_mol in rdkit_mols:
        try:
            sa_scores.append(sascorer.calculateScore(rdkit_mol))
        except Exception as e:
            if verb:
                print("Could not compute SA score for", rdkit_mol, e)
            sa_scores.append(None)
    return np.array(sa_scores)


def compute_fing(rdkit_mols, verb=True):
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


def compute_similarity(rdkit_mols, rdkit_mol1, verb=True):
    print(f"Computing similarity between {len(rdkit_mols)} generated molecules and {rdkit_mol1} seed molecule.")
    fings = compute_fing(rdkit_mols)
    fing1 = compute_fing([rdkit_mol1])[0]
    return np.array([_tanimoto_or_none(f, fing1) for f in fings])


def compute_similarity_fing(fings, fing1):
    return np.array([_tanimoto_or_none(f, fing1) for f in fings])


def predict_binding_score(rdkit_mols, protein, verb=True):
    """Compute binding scores using subprocess to call binding_predictors.py in bind_oracles env."""

    smiles_list = [Chem.MolToSmiles(rdkit_mol) for rdkit_mol in rdkit_mols]
    try:
        # Create the command to run binding_predictors.py in the bind_oracles environment
        predictor_script = resolve_from_project_root(
            "genetic_chemalactica", "oracles", "binding_predictors.py"
        )
        bind_env = os.environ.get("CONDA_ENV_BIND_ORACLES", "bind_oracles")
        cmd = [
            "conda", "run", "-n", bind_env, "python",
            str(predictor_script),
            "--protein", protein
        ] + ["--smiles_lst"] + smiles_list
        
        # Run the command and capture output
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=str(resolve_from_project_root()),
            check=True
        )
        
        # Parse the JSON output
        scores = json.loads(result.stdout.strip())
        return np.array(scores)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running binding predictors for {protein}: {e}")
        print(f"stderr: {e.stderr}")
        return np.array([None] * len(smiles_list))
    except json.JSONDecodeError as e:
        print(f"Error parsing binding predictor output for {protein}: {e}")
        return np.array([None] * len(smiles_list))
    except Exception as e:
        print(f"Unexpected error in binding score computation for {protein}: {e}")
        return np.array([None] * len(smiles_list))