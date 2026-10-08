from __future__ import annotations

from typing import Optional

# RDKit is required
from rdkit import Chem

# torch-geometric is required for SolutionNet
try:
    import torch
    from torch_geometric.data import Data
except Exception as e:  # pragma: no cover
    Data = None  # type: ignore
    torch = None  # type: ignore
    _TORCH_GEOM_IMPORT_ERROR = e
else:
    _TORCH_GEOM_IMPORT_ERROR = None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True)


class GenFeatures:
    """
    This feature generator matches the training code in:
      zsl_cosolvency_project_v2/src/dataset/gen_features.py

    - Node feature dim: 39
    - Edge feature dim: 10
    """
    def __init__(self):
        self.symbols = ['B', 'Br', 'C', 'Ca', 'Cl', 'F', 'H', 'I', 'K', 'Li', 'Mg', 'N', 'Na', 'O', 'P', 'S']
        self.degrees = [0, 1, 2, 3, 4, 5]
        self.formal_charges = [-1, 0, 1]
        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.S,
        ]
        self.hydrogens = [0, 1, 2, 3, 4]
        self.chirality = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                          Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER]
        self.chirality_type = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW]
        self.stereos = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
                        Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]

    def mol_to_data(self, mol: Chem.Mol) -> "Data":
        if _TORCH_GEOM_IMPORT_ERROR is not None:
            raise ImportError(
                "torch and torch_geometric are required for SolutionNet inference. "
                f"Import error: {_TORCH_GEOM_IMPORT_ERROR}"
            )
        if Data is None or torch is None:
            raise ImportError("torch_geometric is not available.")

        data = Data()

        # nodes
        xs = []
        for atom in mol.GetAtoms():
            symbol = [0.0] * len(self.symbols)
            if atom.GetSymbol() in self.symbols:
                symbol[self.symbols.index(atom.GetSymbol())] = 1.0

            degree = [0.0] * len(self.degrees)
            if atom.GetDegree() in self.degrees:
                degree[self.degrees.index(atom.GetDegree())] = 1.0

            formal_charge = atom.GetFormalCharge()
            formal_charge = float(formal_charge) if formal_charge in self.formal_charges else 0.0

            radical_electrons = float(atom.GetNumRadicalElectrons())

            hybridization = [0.0] * len(self.hybridizations)
            if atom.GetHybridization() in self.hybridizations:
                hybridization[self.hybridizations.index(atom.GetHybridization())] = 1.0

            aromaticity = 1.0 if atom.GetIsAromatic() else 0.0

            hydrogens = [0.0] * len(self.hydrogens)
            h = atom.GetTotalNumHs()
            if h in self.hydrogens:
                hydrogens[self.hydrogens.index(h)] = 1.0

            chirality = 1.0 if atom.GetChiralTag() in self.chirality else 0.0

            chirality_type = [0.0] * len(self.chirality_type)
            if atom.GetChiralTag() in self.chirality_type:
                chirality_type[self.chirality_type.index(atom.GetChiralTag())] = 1.0

            node_attr = torch.tensor(symbol + degree + [formal_charge] + [radical_electrons] +
                                     hybridization + [aromaticity] + hydrogens + [chirality] + chirality_type,
                                     dtype=torch.float)
            xs.append(node_attr)

        data.x = torch.stack(xs, dim=0) if xs else torch.zeros((0, 39), dtype=torch.float)

        # edges
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
                             [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

            bond_type = bond.GetBondType()
            single = 1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0
            double = 1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0
            triple = 1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0
            aromatic = 1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0
            conjugation = 1.0 if bond.GetIsConjugated() else 0.0
            ring = 1.0 if bond.IsInRing() else 0.0

            stereo = [0.0] * 4
            st = bond.GetStereo()
            if st in self.stereos:
                stereo[self.stereos.index(st)] = 1.0

            edge_attr = torch.tensor([single, double, triple, aromatic, conjugation, ring] + stereo,
                                     dtype=torch.float)
            edge_attrs += [edge_attr, edge_attr]

        if len(edge_attrs) == 0:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            data.edge_attr = torch.stack(edge_attrs, dim=0)

        return data
def _assign_chiral_tags_compat(mol: Chem.Mol) -> None:
    """
    RDKit 版本兼容：
    - 新版：Chem.AssignAtomChiralTags
    - 旧版：Chem.AssignStereochemistry
    """
    try:
        if hasattr(Chem, "AssignAtomChiralTags"):
            try:
                Chem.AssignAtomChiralTags(mol)
            except TypeError:
                Chem.AssignAtomChiralTags(mol, includeUnassigned=True)
        else:
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    except Exception:
        return

def smiles_to_data(smiles: str) -> "Data":
    if _TORCH_GEOM_IMPORT_ERROR is not None:
        raise ImportError(
            "torch and torch_geometric are required for SolutionNet inference. "
            f"Import error: {_TORCH_GEOM_IMPORT_ERROR}"
        )
    if Data is None or torch is None:
        raise ImportError("torch_geometric is not available.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # canonicalize for consistency
    can = Chem.MolToSmiles(mol, canonical=True)
    mol = Chem.MolFromSmiles(can)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {can}")

    # assign chirality
    # Chem.AssignAtomChiralTags(mol)
    _assign_chiral_tags_compat(mol)
    gen = GenFeatures()
    return gen.mol_to_data(mol)
