from rdkit import Chem


def atom_features_simple(atom: Chem.rdchem.Atom | None) -> int:
    if atom is None:
        return 0
    return min(atom.GetAtomicNum(), 100)


def bond_features_simple(bond: Chem.rdchem.Bond | None) -> int:
    if bond is None:
        return 0
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        return 1
    elif bt == Chem.rdchem.BondType.DOUBLE:
        return 2
    elif bt == Chem.rdchem.BondType.TRIPLE:
        return 3
    elif bt == Chem.rdchem.BondType.AROMATIC:
        return 4
    return 5
