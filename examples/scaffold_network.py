

from rdkit import Chem
from typing import Dict, Set, List, Optional, Any

class NetworkNode:
    """Represents a node in the scaffold network."""
    def __init__(self, mol: Chem.Mol):
        self.mol = mol
        self.parents: List['NetworkNode'] = []
        self.origin_smiles: List[str] = []
        self.non_virtual_origin_smiles: List[str] = []
        self.level: int = 0

    def add_parent(self, parent: 'NetworkNode'):
        if parent not in self.parents:
            self.parents.append(parent)

    def add_origin_smiles(self, smiles: str):
        if smiles not in self.origin_smiles:
            self.origin_smiles.append(smiles)

    def add_non_virtual_origin_smiles(self, smiles: str):
        if smiles not in self.non_virtual_origin_smiles:
            self.non_virtual_origin_smiles.append(smiles)

    def has_non_virtual_origin_smiles(self) -> bool:
        return bool(self.non_virtual_origin_smiles)

    def get_smiles(self) -> str:
        return Chem.MolToSmiles(self.mol, isomericSmiles=True)

class ScaffoldNetwork:
    """Top-level class to organize the NetworkNodes."""
    def __init__(self):
        self.node_map: Dict[int, NetworkNode] = {}
        self.reverse_node_map: Dict[str, int] = {}
        self.smiles_map: Dict[str, NetworkNode] = {}
        self.level_map: Dict[int, Set[NetworkNode]] = {}
        self.node_counter: int = 0

    def add_node(self, node: NetworkNode):
        smiles = node.get_smiles()
        if smiles in self.smiles_map:
            raise ValueError("Node already exists in network")
        idx = self.node_counter
        self.node_map[idx] = node
        self.reverse_node_map[smiles] = idx
        self.smiles_map[smiles] = node
        self.node_counter += 1
        self.update_level_map()

    def remove_node(self, node: NetworkNode):
        smiles = node.get_smiles()
        idx = self.reverse_node_map.get(smiles)
        if idx is None:
            raise ValueError("Node not in network")
        del self.node_map[idx]
        del self.reverse_node_map[smiles]
        del self.smiles_map[smiles]
        level = node.level
        if level in self.level_map:
            self.level_map[level].discard(node)

    def update_level_map(self):
        self.level_map.clear()
        for node in self.node_map.values():
            lvl = node.level
            if lvl not in self.level_map:
                self.level_map[lvl] = set()
            self.level_map[lvl].add(node)

    def contains_molecule(self, mol: Chem.Mol) -> bool:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return smiles in self.smiles_map

    def get_node(self, mol: Chem.Mol) -> Optional[NetworkNode]:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return self.smiles_map.get(smiles)

    def merge_network(self, other: 'ScaffoldNetwork'):
        if not self.node_map:
            for node in other.node_map.values():
                self.add_node(node)
        else:
            new_mols = []
            for node in other.node_map.values():
                if not self.contains_molecule(node.mol):
                    new_node = NetworkNode(node.mol)
                    for s in node.non_virtual_origin_smiles:
                        new_node.add_non_virtual_origin_smiles(s)
                    for s in node.origin_smiles:
                        new_node.add_origin_smiles(s)
                    self.add_node(new_node)
                    new_mols.append(node.mol)
                else:
                    old_node = self.get_node(node.mol)
                    for s in node.origin_smiles:
                        old_node.add_origin_smiles(s)
                    for s in node.non_virtual_origin_smiles:
                        old_node.add_non_virtual_origin_smiles(s)
            for mol in new_mols:
                child = other.get_node(mol)
                for parent in child.parents:
                    if self.contains_molecule(parent.mol):
                        old_parent = self.get_node(parent.mol)
                        old_child = self.get_node(mol)
                        old_child.add_parent(old_parent)
        self.update_level_map()

    def get_matrix(self) -> List[List[int]]:
        size = len(self.node_map)
        idx_list = list(self.node_map.keys())
        idx_map = {idx: i for i, idx in enumerate(idx_list)}
        matrix = [[0] * size for _ in range(size)]
        for idx, node in self.node_map.items():
            row = idx_map[idx]
            for parent in node.parents:
                parent_smiles = parent.get_smiles()
                parent_idx = self.reverse_node_map.get(parent_smiles)
                if parent_idx is not None:
                    col = idx_map[parent_idx]
                    matrix[row][col] = 1
                    matrix[col][row] = 1
        return matrix

    def get_roots(self) -> List[NetworkNode]:
        return [node for node in self.node_map.values() if node.level == 0]

# --- Simple test suggestion ---

if __name__ == "__main__":
    # Create two molecules
    mol1 = Chem.MolFromSmiles("c1ccccc1")  # benzene
    mol2 = Chem.MolFromSmiles("c1ccncc1")  # pyridine

    # Create nodes
    node1 = NetworkNode(mol1)
    node2 = NetworkNode(mol2)
    node2.level = 1
    node2.add_parent(node1)

    # Create network and add nodes
    net = ScaffoldNetwork()
    net.add_node(node1)
    net.add_node(node2)

    # Print adjacency matrix
    print("Adjacency matrix:")
    for row in net.get_matrix():
        print(row)

    # Print root nodes
    print("Root nodes SMILES:")
    for root in net.get_roots():
        print(root.get_smiles())
