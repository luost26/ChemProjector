import pathlib
import shutil
import subprocess
import sys
from collections import Counter

import click
import joblib
import pandas as pd
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from chemprojector.chem.mol import Molecule
from chemprojector.tools.docking import QVinaDockingTask, QVinaOption, QVinaOutput
from chemprojector.utils.tqdm_joblib import tqdm_joblib


class TaskFactory:
    def __init__(
        self,
        receptor_dir: pathlib.Path,
        output_dir: pathlib.Path,
        qvina_path: pathlib.Path,
        obabel_path: pathlib.Path,
        overwrite: bool,
    ) -> None:
        super().__init__()
        self.receptor_dir = receptor_dir
        self.output_dir = output_dir
        self.qvina_path = qvina_path
        self.obabel_path = obabel_path
        self.overwrite = overwrite

    def __call__(
        self,
        receptor_name: str,
        ligand_mol: Molecule,
        output_name: str,
    ) -> QVinaDockingTask:
        rec_desc = OmegaConf.load(self.receptor_dir / receptor_name / "description.yaml")
        qvina_option = QVinaOption(
            center_x=rec_desc.center_of_mass[0],
            center_y=rec_desc.center_of_mass[1],
            center_z=rec_desc.center_of_mass[2],
        )
        output_path = self.output_dir / receptor_name / f"{output_name}.sdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return QVinaDockingTask(
            receptor_path=self.receptor_dir / receptor_name / "receptor.pdbqt",
            ligand_mol=ligand_mol,
            qvina_path=self.qvina_path,
            qvina_option=qvina_option,
            obabel_path=self.obabel_path,
            output_path=output_path,
            overwrite=self.overwrite,
            receptor_name=receptor_name,
        )


@click.command()
@click.option(
    "--pocket2mol-csv",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="data/sbdd/pocket2mol.csv",
)
@click.option(
    "--projected-csv",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="results/sbdd/projected.csv",
)
@click.option("--topk", type=int, default=5)
@click.option("--receptor-dir", type=click.Path(exists=True, path_type=pathlib.Path), default="data/sbdd/receptors")
@click.option("--output-dir", type=click.Path(path_type=pathlib.Path), default="results/sbdd/docking")
@click.option("--n-jobs", type=int, default=joblib.cpu_count() // 2)
@click.option(
    "--qvina-path",
    type=click.Path(dir_okay=False, executable=True, path_type=pathlib.Path),
    default="bin/qvina2.1",
)
@click.option(
    "--obabel-path",
    type=click.Path(exists=True, dir_okay=False, executable=True, path_type=pathlib.Path),
    default=shutil.which("obabel"),
)
def main(
    pocket2mol_csv: pathlib.Path,
    projected_csv: pathlib.Path,
    topk: int,
    receptor_dir: pathlib.Path,
    output_dir: pathlib.Path,
    n_jobs: int,
    qvina_path: pathlib.Path,
    obabel_path: pathlib.Path,
):
    if not qvina_path.exists():
        if sys.platform == "linux":
            print(f"QVina2.1 not found at {qvina_path}, attempting to download...")
            subprocess.run(
                "wget -O bin/qvina2.1 https://github.com/QVina/qvina/raw/master/bin/qvina2.1",
                shell=True,
                check=True,
            )
            subprocess.run("chmod +x bin/qvina2.1", shell=True, check=True)
        else:
            raise FileNotFoundError(f"QVina2.1 not found at {qvina_path}")

    df_pocket2mol = pd.read_csv(pocket2mol_csv)
    df_pocket2mol = df_pocket2mol.rename(columns={"smiles": "smiles_pocket2mol", "vina": "vina_pocket2mol"})
    df_projected = pd.read_csv(projected_csv)
    df_projected = (
        df_projected.groupby("target")
        .apply(lambda x: x.nlargest(topk, "score"))
        .reset_index(drop=True)
        .rename(columns={"target": "smiles_pocket2mol"})
    )

    df = pd.merge(df_pocket2mol, df_projected, on="smiles_pocket2mol", how="inner")

    counter = Counter[str]()
    tasks: list[QVinaDockingTask] = []
    overwrite = False
    if output_dir.exists():
        overwrite = click.confirm(f"{output_dir} already exists. Overwrite?")
    factory = TaskFactory(
        receptor_dir=receptor_dir,
        output_dir=output_dir,
        qvina_path=qvina_path,
        obabel_path=obabel_path,
        overwrite=overwrite,
    )

    for _, row in tqdm(df.iterrows(), desc="Generating tasks", total=len(df)):
        receptor_name = row["receptor"]
        mol = Molecule(row["smiles"])
        tasks.append(
            factory(
                receptor_name=receptor_name,
                ligand_mol=mol,
                output_name=f"{counter[receptor_name]}",
            )
        )
        counter[receptor_name] += 1

    with tqdm_joblib(tqdm(desc="Docking", total=len(tasks))):
        outputs: list[QVinaOutput | None] = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(task)() for task in tasks)

    results: list[dict] = []
    for output in outputs:
        if output is None:
            continue
        results.append(
            {
                "receptor": output.receptor_name,
                "smiles": output.smiles,
                "vina": output.affinity,
                "path": output.path,
            }
        )
    df_results = pd.merge(df, pd.DataFrame(results), on=["receptor", "smiles"], how="inner")
    df_results.to_csv(output_dir / "docking.csv", index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
