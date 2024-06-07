import pathlib

import click
import pandas as pd


@click.command()
@click.option("--result-dir", type=click.Path(exists=True, path_type=pathlib.Path), default="results/sbdd/docking")
def main(result_dir: pathlib.Path):
    df = pd.read_csv(result_dir / "docking.csv")

    # Select five most similar analogs
    df = df.groupby(["receptor", "smiles_pocket2mol"]).apply(lambda x: x.nlargest(5, "score")).reset_index(drop=True)

    # Choose the analog with best Vina score among the five most similar analogs
    df = df.loc[df.groupby(["receptor", "smiles_pocket2mol"]).idxmin()["vina"]]

    # Summary by receptor
    summary = df.groupby("receptor").agg(
        {
            "vina_pocket2mol": ["mean"],
            "vina": ["mean"],
            "score": ["mean"],
            "scf_sim": ["mean"],
            "pharm2d_sim": ["mean"],
            "rdkit_sim": ["mean"],
        }
    )
    print(summary)


if __name__ == "__main__":
    main()
