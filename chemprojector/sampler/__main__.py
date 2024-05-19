import os

import click

from chemprojector.chem.mol import Molecule, read_mol_file

from .parallel import run_parallel_sampling, sampler_options


def _input_mols_option(p):
    return list(read_mol_file(p))


@click.command()
@click.option("--input", "-i", type=_input_mols_option, required=True)
@click.option("--output", "-o", type=click.Path(exists=False), required=True)
@sampler_options
def main(
    input: list[Molecule],
    output: os.PathLike,
    model_path: os.PathLike,
    rxn_matrix_path: os.PathLike,
    fpindex_path: os.PathLike,
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
