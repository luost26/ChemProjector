import multiprocessing as mp
import os
import pathlib
import pickle
import subprocess
from multiprocessing import synchronize as sync
from typing import TypeAlias

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from chemprojector.chem.fpindex import FingerprintIndex
from chemprojector.chem.matrix import ReactantReactionMatrix
from chemprojector.chem.mol import FingerprintOption, Molecule
from chemprojector.models.chemprojector import ChemProjector

from .state_pool import StatePool, TimeLimit

TaskQueueType: TypeAlias = "mp.JoinableQueue[Molecule | None]"
ResultQueueType: TypeAlias = "mp.Queue[tuple[Molecule, pd.DataFrame]]"


class Worker(mp.Process):
    def __init__(
        self,
        fpindex_path: pathlib.Path,
        rxn_matrix_path: pathlib.Path,
        model_path: pathlib.Path,
        task_queue: TaskQueueType,
        result_queue: ResultQueueType,
        gpu_id: str,
        gpu_lock: sync.Lock,
        state_pool_opt: dict | None = None,
        max_evolve_steps: int = 12,
        max_results: int = 100,
        time_limit: int = 120,
    ):
        super().__init__()
        self._fpindex_path = fpindex_path
        self._rxn_matrix_path = rxn_matrix_path
        self._model_path = model_path
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._gpu_id = gpu_id
        self._gpu_lock = gpu_lock

        self._state_pool_opt = state_pool_opt or {}
        self._max_evolve_steps = max_evolve_steps
        self._max_results = max_results
        self._time_limit = time_limit

    def run(self) -> None:
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
        os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu_id

        self._fpindex: FingerprintIndex = pickle.load(open(self._fpindex_path, "rb"))
        self._rxn_matrix: ReactantReactionMatrix = pickle.load(open(self._rxn_matrix_path, "rb"))

        ckpt = torch.load(self._model_path, map_location="cpu", weights_only=False)
        config = OmegaConf.create(ckpt["hyper_parameters"]["config"])
        model = ChemProjector(config.model).to("cuda")
        model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
        model.eval()
        self._model = model

        try:
            while True:
                next_task = self._task_queue.get()
                if next_task is None:
                    self._task_queue.task_done()
                    break
                result_df = self.process(next_task)
                self._task_queue.task_done()
                self._result_queue.put((next_task, result_df))
                if len(result_df) == 0:
                    print(f"{self.name}: No results for {next_task.smiles}")
                else:
                    max_sim = result_df["score"].max()
                    print(f"{self.name}: {max_sim:.3f} {next_task.smiles}")
        except KeyboardInterrupt:
            print(f"{self.name}: Exiting due to KeyboardInterrupt")
            return

    def process(self, mol: Molecule):
        sampler = StatePool(
            fpindex=self._fpindex,
            rxn_matrix=self._rxn_matrix,
            mol=mol,
            model=self._model,
            **self._state_pool_opt,
        )
        tl = TimeLimit(self._time_limit)
        for _ in range(self._max_evolve_steps):
            sampler.evolve(gpu_lock=self._gpu_lock, show_pbar=False, time_limit=tl)
            max_sim = max(
                [
                    p.molecule.sim(mol, FingerprintOption.morgan_for_tanimoto_similarity())
                    for p in sampler.get_products()
                ]
                or [-1]
            )
            if max_sim == 1.0:
                break

        df = sampler.get_dataframe()[: self._max_results]
        return df


class WorkerPool:
    def __init__(
        self,
        gpu_ids: list[int | str],
        num_workers_per_gpu: int,
        task_qsize: int,
        result_qsize: int,
        **worker_opt,
    ) -> None:
        super().__init__()
        self._task_queue: TaskQueueType = mp.JoinableQueue(task_qsize)
        self._result_queue: ResultQueueType = mp.Queue(result_qsize)
        self._gpu_ids = [str(d) for d in gpu_ids]
        self._gpu_locks = [mp.Lock() for _ in gpu_ids]
        num_gpus = len(gpu_ids)
        num_workers = num_workers_per_gpu * num_gpus
        self._workers = [
            Worker(
                task_queue=self._task_queue,
                result_queue=self._result_queue,
                gpu_id=self._gpu_ids[i % num_gpus],
                gpu_lock=self._gpu_locks[i % num_gpus],
                **worker_opt,
            )
            for i in range(num_workers)
        ]

        for w in self._workers:
            w.start()

    def submit(self, task: Molecule, block: bool = True, timeout: float | None = None):
        self._task_queue.put(task, block=block, timeout=timeout)

    def fetch(self, block: bool = True, timeout: float | None = None):
        return self._result_queue.get(block=block, timeout=timeout)

    def kill(self):
        for w in self._workers:
            w.kill()
        self._result_queue.close()
        self._task_queue.close()

    def end(self):
        for _ in self._workers:
            self._task_queue.put(None)
        self._task_queue.join()
        for w in tqdm(self._workers, desc="Terminating"):
            w.terminate()
        self._result_queue.close()
        self._task_queue.close()


def _count_gpus():
    return int(
        subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l", shell=True, text=True
        ).strip()
    )


def run_parallel_sampling(
    input: list[Molecule],
    output: pathlib.Path,
    model_path: pathlib.Path,
    rxn_matrix_path: pathlib.Path,
    fpindex_path: pathlib.Path,
    search_width: int = 24,
    exhaustiveness: int = 64,
    num_gpus: int = -1,
    num_workers_per_gpu: int = 2,
    task_qsize: int = 0,
    result_qsize: int = 0,
    time_limit: int = 180,
) -> None:
    num_gpus = num_gpus if num_gpus > 0 else _count_gpus()
    pool = WorkerPool(
        gpu_ids=list(range(num_gpus)),
        num_workers_per_gpu=num_workers_per_gpu,
        task_qsize=task_qsize,
        result_qsize=result_qsize,
        fpindex_path=fpindex_path,
        rxn_matrix_path=rxn_matrix_path,
        model_path=model_path,
        state_pool_opt={
            "factor": search_width,
            "max_active_states": exhaustiveness,
        },
        time_limit=time_limit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    total = len(input)
    for mol in input:
        pool.submit(mol)

    df_all: list[pd.DataFrame] = []
    with open(output, "w") as f:
        for _ in tqdm(range(total)):
            _, df = pool.fetch()
            if len(df) == 0:
                continue
            df.to_csv(f, float_format="%.3f", index=False, header=f.tell() == 0)
            df_all.append(df)

    df_merge = pd.concat(df_all, ignore_index=True)
    summary = df_merge.loc[df_merge.groupby("target").idxmax()["score"]].select_dtypes(include="number").sum() / total
    desc = pd.Series(
        {
            "score": "Tanimoto similarity",
            "scf_sim": "Scaffold similarity",
            "pharm2d_sim": "Gobbi2D Pharmacophore similarity",
            "rdkit_sim": "RDKit fingerprint similarity",
            "num_steps": "Average number of synthesis steps",
        }
    )
    print(pd.DataFrame({"result": summary, "description": desc}))

    count_success = len(df_merge["target"].unique())
    print(f"Success rate: {count_success}/{total} = {count_success / total:.3f}")

    recons_targets: set[str] = set()
    for _, row in df_merge.iterrows():
        if row["score"] == 1.0:
            mol_target = Molecule(row["target"])
            mol_recons = Molecule(row["smiles"])
            if mol_recons.csmiles == mol_target.csmiles:
                recons_targets.add(row["target"])
    count_recons = len(recons_targets)
    print(f"Reconstruction rate: {count_recons}/{total} = {count_recons / total:.3f}")

    pool.end()
