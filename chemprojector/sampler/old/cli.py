import pathlib

import click

from chemprojector.chem.mol import Molecule, read_mol_file

from .parallel import run_parallel_sampling


def _input_mols_option(p):
    return list(read_mol_file(p))


@click.command()
@click.option("--input", "-i", type=_input_mols_option, required=True)
@click.option("--output", "-o", type=click.Path(exists=False, path_type=pathlib.Path), required=True)
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="data/trained_weights/old/default.ckpt",
)
@click.option(
    "--rxn-matrix-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="data/processed/all/matrix.pkl",
)
@click.option(
    "--fpindex-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default="data/processed/all/fpindex.pkl",
)
@click.option("--search-width", type=int, default=24)
@click.option("--exhaustiveness", type=int, default=64)
@click.option("--num-gpus", type=int, default=4)
@click.option("--num-workers-per-gpu", type=int, default=1)
@click.option("--task-qsize", type=int, default=0)
@click.option("--result-qsize", type=int, default=0)
@click.option("--time-limit", type=int, default=180)
def main(
    input: list[Molecule],
    output: pathlib.Path,
    model_path: pathlib.Path,
    rxn_matrix_path: pathlib.Path,
    fpindex_path: pathlib.Path,
    search_width: int,
    exhaustiveness: int,
    num_gpus: int,
    num_workers_per_gpu: int,
    task_qsize: int,
    result_qsize: int,
    time_limit: int,
):
    run_parallel_sampling(
        input=input,
        output=output,
        model_path=model_path,
        rxn_matrix_path=rxn_matrix_path,
        fpindex_path=fpindex_path,
        search_width=search_width,
        exhaustiveness=exhaustiveness,
        num_gpus=num_gpus,
        num_workers_per_gpu=num_workers_per_gpu,
        task_qsize=task_qsize,
        result_qsize=result_qsize,
        time_limit=time_limit,
    )


if __name__ == "__main__":
    main()
