import pathlib

import click

from chemprojector.chem.matrix import create_reactant_reaction_matrix_cache

_default_sdf_path = pathlib.Path("data/Enamine_Rush-Delivery_Building_Blocks-US_223244cmpd_20231001.sdf")


@click.command()
@click.option(
    "--reactant",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=_default_sdf_path,
)
@click.option(
    "--reaction",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=pathlib.Path("data/reaction_templates_hb.txt"),
)
@click.option("--out", type=click.Path(path_type=pathlib.Path), default=pathlib.Path("data/processed/all/matrix.pkl"))
def matrix(reactant: pathlib.Path, reaction: pathlib.Path, out: pathlib.Path):
    if out.exists():
        click.confirm(f"{out} already exists. Overwrite?", abort=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    m = create_reactant_reaction_matrix_cache(
        reactant_path=reactant,
        reaction_path=reaction,
        cache_path=out,
    )
    print(f"Number of reactants: {len(m.reactants)}")
    print(f"Number of reactions: {len(m.reactions)}")
    print(f"Saved matrix to {out}")


if __name__ == "__main__":
    matrix()
