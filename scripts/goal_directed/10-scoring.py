import pathlib
from collections.abc import Sequence

import click
import joblib
import pandas as pd
from tqdm.auto import tqdm

from chemprojector.tools.guacamol import GuacamolScoringFunction
from chemprojector.utils.tqdm_joblib import tqdm_joblib


def _evaulate_property_scores(property_list: Sequence[str], smiles_list: Sequence[str]) -> list[float]:
    with tqdm_joblib(tqdm(desc="Evaluating properties", total=len(smiles_list))):
        return joblib.Parallel(n_jobs=joblib.cpu_count() // 2, batch_size=512)(
            joblib.delayed(GuacamolScoringFunction.get_preset(prop))(smiles)
            for prop, smiles in zip(property_list, smiles_list)
        )


@click.command()
@click.option(
    "--source-csv",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="data/goal_directed/goal_hard_cwo.csv",
)
@click.option(
    "--projected-csv",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="results/goal_directed/projected.csv",
)
def main(source_csv: pathlib.Path, projected_csv: pathlib.Path):
    df_source = pd.read_csv(source_csv).rename(columns={"SMILES": "original_smiles"})
    df_source = (
        # Remove non-generated or synthesizable molecules (taken from Chembl)
        df_source[(df_source["method"] != "best_from_chembl") & (df_source["tb_synthesizability"] <= 0)]
        .reset_index(drop=True)
        .groupby(["method", "property"])
        .apply(lambda x: x.nlargest(100, "objective"))  # Select best 100 molecules for each property
        .reset_index(drop=True)
    )

    df_projected = pd.read_csv(projected_csv).rename(columns={"target": "original_smiles", "score": "morgan_sim"})
    # Select 10 most similar analogs for each original molecule
    df_projected = (
        df_projected.groupby("original_smiles").apply(lambda x: x.nlargest(10, "morgan_sim")).reset_index(drop=True)
    )

    df_merge = pd.merge(df_source, df_projected, on="original_smiles", how="left").dropna().reset_index(drop=True)

    # Evaluate objective scores
    df_merge["score_original"] = _evaulate_property_scores(df_merge["property"], df_merge["original_smiles"])
    df_merge["score_projected"] = _evaulate_property_scores(df_merge["property"], df_merge["smiles"])

    # For each original molecule, select the analog with the highest objective score
    df_out = (
        df_merge.groupby("original_smiles")
        .apply(lambda x: x.nlargest(1, "score_projected"))
        .reset_index(drop=True)
        .groupby("property")
        .apply(lambda x: x.nlargest(100, "score_projected"))
        .reset_index(drop=True)
    )

    summary = df_out.groupby(["property"]).agg(
        {
            "smiles": ["count"],
            "morgan_sim": ["mean"],
            "scf_sim": ["mean"],
            "pharm2d_sim": ["mean"],
            "score_original": ["mean", "max"],
            "score_projected": ["mean", "max"],
        }
    )
    print(summary)


if __name__ == "__main__":
    main()
