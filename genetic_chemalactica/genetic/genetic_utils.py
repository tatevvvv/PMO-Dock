from typing import List
import random
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors

# Disable RDKit logs
RDLogger.DisableLog("rdApp.*")

# import safe


def get_morgan_fingerprint(mol):
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return None


def get_maccs_fingerprint(mol):
    return MACCSkeys.GenMACCSKeys(mol)


def tanimoto_dist_func(fing1, fing2, fingerprint: str = "morgan"):
    return DataStructs.TanimotoSimilarity(
        fing1 if fingerprint == "morgan" else fing1,
        fing2 if fingerprint == "morgan" else fing2,
    )


def canonicalize(smiles):
    """Return canonical SMILES, or None if the string is empty or does not parse."""
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


class Entry:

    def __init__(
        self,
        smiles,
        score,
        **kwargs
    ):
        self.mol = canonicalize(smiles)
        if self.mol is None:
            raise ValueError(f"invalid SMILES for Entry: {smiles!r}")
        self.rdkit_mol = Chem.MolFromSmiles(self.mol)
        if self.rdkit_mol is None:
            raise ValueError(f"RDKit could not re-parse canonical SMILES: {self.mol!r}")
        self.score = score
        self.fingerprint = get_morgan_fingerprint(self.rdkit_mol)
        if self.fingerprint is None:
            raise ValueError(f"could not fingerprint mol: {self.mol!r}")
        self.add_props = kwargs

    def __eq__(self, other):
        return self.mol == other.mol

    def __lt__(self, other):
        if self.score == other.score:
            return self.mol < other.mol
        return self.score < other.score

    def __str__(self):
        return (
            f"Entry: mol={self.mol}, "
            f"score={round(self.score, 4) if self.score is not None else 'none'}"
        )

    def __repr__(self):
        return str(self)
    
    def __hash__(self):
        return hash(self.mol)


class Pool:

    def __init__(self, size):
        self.size = size
        self.entries: List[Entry] = []

    # def random_dump(self, num):
    #     for _ in range(num):
    #         rand_ind = random.randint(0, num - 1)
    #         self.molecule_entries.pop(rand_ind)
    #     print(f"Dump {num} random elements from pool, num pool mols {len(self)}")

    def add(self, entries: List, diversity_score=1.0):
        assert type(entries) == list
        self.entries.extend(entries)
        self.entries.sort(reverse=True)

        # remove doublicates
        new_entries = []
        for entry in self.entries:
            insert = True
            for e in new_entries:
                if (
                    entry == e
                    or tanimoto_dist_func(
                        entry.fingerprint, e.fingerprint
                    )
                    > diversity_score
                ):
                    insert = False
                    break
            if insert:
                new_entries.append(entry)

        self.entries = new_entries[: min(len(new_entries), self.size)]

    def random_subset(self, subset_size):
        rand_inds = np.random.permutation(len(self.entries))
        rand_inds = rand_inds[:subset_size]
        return [self.entries[i] for i in rand_inds]

    def __len__(self):
        return len(self.entries)