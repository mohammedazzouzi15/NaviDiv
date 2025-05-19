import concurrent.futures
from collections import Counter

from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage


def check_mols_has_bond_not_in_ring(mol):
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            return True
    return False


def Fragment_rings(mol):
    frags = set()
    smarts_to_check = "[R0]"
    if mol is None:
        return frags
    Chem.Kekulize(mol, clearAromaticFlags=True)
    smarts_mol = Chem.MolFromSmarts(smarts_to_check)
    atoms_in_rings = []
    atoms_in_multiple_rings = set()
    ring_info = mol.GetRingInfo()
    for atom_ring in ring_info.AtomRings():
        atoms_in_rings.extend(atom_ring)
    counter = Counter(atoms_in_rings)
    for atom_ring in ring_info.AtomRings():
        atoms_to_keep = set()
        atoms_to_delete = set()
        for atom_idx in atom_ring:
            if counter[atom_idx] > 1:
                atoms_in_multiple_rings.add(atom_idx)
            if atom_idx in atoms_in_multiple_rings:
                atoms_to_keep.add(atom_idx)
            atoms_to_delete.add(atom_idx)
        atoms_to_delete.difference_update(atoms_to_keep)
        mol_copy = Chem.RWMol(mol)
        mol_copy.BeginBatchEdit()
        for atom in mol_copy.GetAtoms():
            if atom.GetIdx() in atoms_to_delete:
                mol_copy.RemoveAtom(atom.GetIdx())
        mol_copy.CommitBatchEdit()
        frags.update(
            Chem.GetMolFrags(mol_copy, asMols=True, sanitizeFrags=True)
        )
    #frags = [frag for frag in frags if not frag.HasSubstructMatch(smarts_mol)]
    #frags = [
    #    frag for frag in frags if not check_mols_has_bond_not_in_ring(frag)
    #]
    return frags


def fragment_list(frags, mols_smiles):
    for frag in frags:
        frags_n = Fragment_rings(frag)
        frag_smiles = [
            Chem.MolToSmiles(frag) for frag in frags_n
        ]
        frag_smiles = set(frag_smiles)
        mols_smiles.update(frag_smiles)
    return mols_smiles


def fragment_mols_fused_ring(mol):
    mols_smiles = set()
    frags = Fragment_rings(mol)
    frag_smiles = [
        Chem.MolToSmiles(frag) for frag in frags
    ]
    frag_smiles = set(frag_smiles)
    frags = [Chem.MolFromSmiles(frag) for frag in frag_smiles]
    mols_smiles.update(frag_smiles)
    old_mols_smiles_len = 0
    while old_mols_smiles_len != len(mols_smiles):
        old_mols_smiles_len = len(mols_smiles)
        mols_smiles = fragment_list(frags, mols_smiles)
        frags = [Chem.MolFromSmiles(frag) for frag in mols_smiles]
    return [Chem.MolFromSmiles(smi) for smi in mols_smiles]


def process_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return fragment_mols_fused_ring(mol)
    return []


if __name__ == "__main__":
    smiles_strings = [
        "[#6]1:[#6]:[#6]:[#6]2:[#6](:[#6]:1):[#6]:[#6]:[#6]1:[#6]:2-[#6]2=[#6]-1-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#6]-[#6]-2",
    ]

    mols_smiles = set()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_smiles, smiles_strings))

    for result in results:
        for mol in result:
            mols_smiles.add(mol)
    params =Chem.rdmolfiles.SmilesWriteParams()
    params.isomericSmiles = False
    params.allBondsExplicit = True
    params.canonical = True
    smarts = [
        Chem.MolToSmarts(mol,params) for mol in mols_smiles
    ]
    import pandas as pd
    df_smarts = pd.DataFrame(
        {"Fragment": smarts}
    )   
    df_smarts.to_csv("fragment_counts_one.csv", index=False)

    if len(mols_smiles) > 0:
        img = MolsToGridImage(
            mols_smiles,
            subImgSize=(300, 200),
        )
        img.save("test_prune.png")
    else:
        print("No fragments found")
