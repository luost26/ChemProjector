import pathlib
import pickle
import random

import click
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm

from .fpindex import (
    FingerprintIndex,
    compute_fingerprints,
    create_fingerprint_index_cache,
)
from .matrix import ReactantReactionMatrix, create_reactant_reaction_matrix_cache
from .mol import FingerprintOption, Molecule, read_mol_file, write_to_smi
from .reaction import ReactionContainer, read_reaction_file
from .stack import Stack, create_stack

_default_sdf_path = pathlib.Path("data/Enamine_Rush-Delivery_Building_Blocks-US_223244cmpd_20231001.sdf")


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--reactant",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=_default_sdf_path,
)
@click.option(
    "--reaction",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=pathlib.Path("data/hartenfeller_button.txt"),
)
@click.option("--out", type=click.Path(path_type=pathlib.Path), default=pathlib.Path("data/processed/all/matrix.pkl"))
def matrix(reactant: pathlib.Path, reaction: pathlib.Path, out: pathlib.Path):
    if out.exists():
        click.confirm(f"{out} already exists. Overwrite?", abort=True)

    m = create_reactant_reaction_matrix_cache(
        reactant_path=reactant,
        reaction_path=reaction,
        cache_path=out,
    )
    print(f"Number of reactants: {len(m.reactants)}")
    print(f"Number of reactions: {len(m.reactions)}")
    print(f"Saved matrix to {out}")


@cli.command()
@click.option(
    "--model-config",
    type=OmegaConf.load,
    required=True,
)
@click.option(
    "--reactant",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=_default_sdf_path,
)
@click.option(
    "--reaction",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=pathlib.Path("data/hartenfeller_button.txt"),
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=pathlib.Path),
    default=pathlib.Path("data/processed/split"),
)
@click.option("--seed", type=int, default=42)
@click.option("--num-clusters", type=int, default=100)
def matrix_split(
    model_config: DictConfig,
    reactant: pathlib.Path,
    reaction: pathlib.Path,
    out_dir: pathlib.Path,
    seed: int,
    num_clusters: int,
):
    if out_dir.exists():
        click.confirm(f"{out_dir} already exists. Overwrite?", abort=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Cluster reactants
    mols = list(read_mol_file(reactant))
    fp = compute_fingerprints(mols, fp_option=FingerprintOption.morgan_for_building_blocks())
    clu = MiniBatchKMeans(n_clusters=num_clusters, random_state=seed, verbose=1, batch_size=10000).fit(fp)

    mols_train: list[Molecule] = []
    mols_test: list[Molecule] = []
    for mol, label in zip(mols, clu.labels_):
        if label == 0:
            mols_test.append(mol)
        else:
            mols_train.append(mol)
    print("Number of molecules in training set:", len(mols_train))
    print("Number of molecules in test set:", len(mols_test))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_to_smi(out_dir / "reactant_train.smi", mols_train)
    write_to_smi(out_dir / "reactant_test.smi", mols_test)

    rxns = ReactionContainer(read_reaction_file(reaction))
    matrix_train = ReactantReactionMatrix(mols_train, rxns)
    matrix_test = ReactantReactionMatrix(mols_test, rxns)
    with open(out_dir / "matrix_train.pkl", "wb") as f:
        pickle.dump(matrix_train, f)
    with open(out_dir / "matrix_test.pkl", "wb") as f:
        pickle.dump(matrix_test, f)

    fp_option = FingerprintOption(**model_config.chem.fp_option)
    fpindex_train = FingerprintIndex(molecules=mols_train, fp_option=fp_option)
    with open(out_dir / "fpindex_train.pkl", "wb") as f:
        pickle.dump(fpindex_train, f)


@cli.command()
@click.option(
    "--split-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=pathlib.Path),
    default="data/processed/split",
)
@click.option("--num-stacks", type=int, default=1000)
@click.option("--stack-max-num-reactions", type=int, default=5)
@click.option("--stack-max-num-atoms", type=int, default=80)
@click.option("--seed", type=int, default=42)
def test_stacks(
    split_dir: pathlib.Path,
    num_stacks: int,
    stack_max_num_reactions: int,
    stack_max_num_atoms: int,
    seed: int,
):
    if (split_dir / "stacks_test.pkl").exists():
        click.confirm(f"{split_dir / 'stacks_test.pkl'} already exists. Overwrite?", abort=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    with open(split_dir / "matrix_test.pkl", "rb") as f:
        matrix_test = pickle.load(f)

    stacks: list[Stack] = []
    visited: set[str] = set()
    num_collisions: int = 0
    with tqdm(total=num_stacks, desc="Stacks") as pbar:
        while (i := len(stacks)) < num_stacks:
            stack = create_stack(
                matrix_test,
                max_num_reactions=(i % stack_max_num_reactions) + 1,
                max_num_atoms=stack_max_num_atoms,
            )
            prod = next(iter(stack.get_top()))
            if prod.csmiles in visited:
                num_collisions += 1
                continue
            stacks.append(stack)
            visited.add(prod.csmiles)
            pbar.update(1)
    with open(split_dir / "stacks_test.pkl", "wb") as f:
        pickle.dump(stacks, f)
    print(f"Number of stacks: {len(stacks)}")
    print(f"Number of collisions: {num_collisions}")

    smiles: list[str] = ["smiles,synthesis"]
    for stack in stacks:
        mol = next(iter(stack.get_top()))
        smiles.append(f"{mol.smiles},{stack.get_action_string()}")
    (split_dir / "stacks_test.csv").write_text("\n".join(smiles))


@cli.command()
@click.option(
    "--model-config",
    type=OmegaConf.load,
    required=True,
)
@click.option(
    "--molecule",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=_default_sdf_path,
)
@click.option("--out", type=click.Path(path_type=pathlib.Path), default=pathlib.Path("data/processed/all/fpindex.pkl"))
def fpindex(model_config: DictConfig, molecule: pathlib.Path, out: pathlib.Path):
    if out.exists():
        click.confirm(f"{out} already exists. Overwrite?", abort=True)

    fp_option = FingerprintOption(**model_config.chem.fp_option)
    fpindex = create_fingerprint_index_cache(
        molecule_path=molecule,
        cache_path=out,
        fp_option=fp_option,
    )
    print(f"Number of molecules: {len(fpindex.molecules)}")
    print(f"Saved index to {out}")


if __name__ == "__main__":
    cli()
