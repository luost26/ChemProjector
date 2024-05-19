import enum
import os
import pickle
import random
from collections.abc import Sequence
from typing import TypedDict, cast

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from chemprojector.chem.fpindex import FingerprintIndex
from chemprojector.chem.matrix import ReactantReactionMatrix
from chemprojector.chem.mol import Molecule
from chemprojector.chem.reaction import Reaction
from chemprojector.chem.stack import Stack, create_stack
from chemprojector.utils.image import draw_text, make_grid
from chemprojector.utils.train import worker_init_fn

from .collate import (
    apply_collate,
    collate_1d_features,
    collate_2d_tokens,
    collate_padding_masks,
    collate_tokens,
)


class TokenType(enum.IntEnum):
    END = 0
    START = 1
    REACTION = 2
    REACTANT = 3


class ProjectionData(TypedDict, total=False):
    # Encoder
    atoms: torch.Tensor
    bonds: torch.Tensor
    atom_padding_mask: torch.Tensor
    # Decoder
    token_types: torch.Tensor
    rxn_indices: torch.Tensor
    reactant_fps: torch.Tensor
    token_padding_mask: torch.Tensor
    # Auxilliary
    mol_seq: Sequence[Molecule]
    rxn_seq: Sequence[Reaction | None]


class ProjectionBatch(TypedDict, total=False):
    # Encoder
    atoms: torch.Tensor
    bonds: torch.Tensor
    atom_padding_mask: torch.Tensor
    # Decoder
    token_types: torch.Tensor
    rxn_indices: torch.Tensor
    reactant_fps: torch.Tensor
    token_padding_mask: torch.Tensor
    # Auxilliary
    mol_seq: Sequence[Sequence[Molecule]]
    rxn_seq: Sequence[Sequence[Reaction | None]]


def featurize_stack_actions(
    mol_idx_seq: Sequence[int | None],
    rxn_idx_seq: Sequence[int | None],
    end_token: bool,
    fpindex: FingerprintIndex,
) -> dict[str, torch.Tensor]:
    seq_len = len(mol_idx_seq) + 1  # Plus START token
    if end_token:
        seq_len += 1
    fp_dim = fpindex.fp_option.dim
    feats = {
        "token_types": torch.zeros([seq_len], dtype=torch.long),
        "rxn_indices": torch.zeros([seq_len], dtype=torch.long),
        "reactant_fps": torch.zeros([seq_len, fp_dim], dtype=torch.float),
        "token_padding_mask": torch.zeros([seq_len], dtype=torch.bool),
    }
    feats["token_types"][0] = TokenType.START
    for i, (mol_idx, rxn_idx) in enumerate(zip(mol_idx_seq, rxn_idx_seq), start=1):
        if rxn_idx is not None:
            feats["token_types"][i] = TokenType.REACTION
            feats["rxn_indices"][i] = rxn_idx
        elif mol_idx is not None:
            feats["token_types"][i] = TokenType.REACTANT
            _, mol_fp = fpindex[mol_idx]
            feats["reactant_fps"][i] = torch.from_numpy(mol_fp)
    return feats


def featurize_stack(stack: Stack, end_token: bool, fpindex: FingerprintIndex) -> dict[str, torch.Tensor]:
    return featurize_stack_actions(
        mol_idx_seq=stack.get_mol_idx_seq(),
        rxn_idx_seq=stack.get_rxn_idx_seq(),
        end_token=end_token,
        fpindex=fpindex,
    )


def create_data(
    product: Molecule,
    mol_seq: Sequence[Molecule],
    mol_idx_seq: Sequence[int | None],
    rxn_seq: Sequence[Reaction | None],
    rxn_idx_seq: Sequence[int | None],
    fpindex: FingerprintIndex,
):
    atom_f, bond_f = product.featurize_simple()
    stack_feats = featurize_stack_actions(
        mol_idx_seq=mol_idx_seq,
        rxn_idx_seq=rxn_idx_seq,
        end_token=True,
        fpindex=fpindex,
    )

    data: "ProjectionData" = {
        "mol_seq": mol_seq,
        "rxn_seq": rxn_seq,
        "atoms": atom_f,
        "bonds": bond_f,
        "atom_padding_mask": torch.zeros([atom_f.size(0)], dtype=torch.bool),
        "token_types": stack_feats["token_types"],
        "rxn_indices": stack_feats["rxn_indices"],
        "reactant_fps": stack_feats["reactant_fps"],
        "token_padding_mask": stack_feats["token_padding_mask"],
    }
    return data


class Collater:
    def __init__(self, max_datapoints: int | None = None, max_num_atoms: int = 96, max_num_tokens: int = 24):
        super().__init__()
        self.max_datapoints = max_datapoints
        self.max_num_atoms = max_num_atoms
        self.max_num_tokens = max_num_tokens

        self.spec_atoms = {
            "atoms": collate_tokens,
            "bonds": collate_2d_tokens,
            "atom_padding_mask": collate_padding_masks,
        }
        self.spec_tokens = {
            "token_types": collate_tokens,
            "rxn_indices": collate_tokens,
            "reactant_fps": collate_1d_features,
            "token_padding_mask": collate_padding_masks,
        }

    def __call__(self, data_list: list[list[ProjectionData]]) -> ProjectionBatch:
        flat = [d for subl in data_list for d in subl]
        if self.max_datapoints is not None:
            random.shuffle(flat)
            flat = flat[: self.max_datapoints]
            if len(flat) < self.max_datapoints:
                flat += random.choices(flat, k=self.max_datapoints - len(flat))
        flat_t = cast(list[dict[str, torch.Tensor]], flat)
        batch = {
            **apply_collate(self.spec_atoms, flat_t, max_size=self.max_num_atoms),
            **apply_collate(self.spec_tokens, flat_t, max_size=self.max_num_tokens),
            "mol_seq": [d["mol_seq"] for d in flat],
            "rxn_seq": [d["rxn_seq"] for d in flat],
        }
        return cast(ProjectionBatch, batch)


class ProjectionDataset(Dataset):
    def __init__(
        self,
        reaction_matrix: ReactantReactionMatrix,
        fpindex: FingerprintIndex,
        max_num_atoms: int = 80,
        max_num_reactions: int = 5,
        virtual_length: int = 65536,
        init_stack_weighted_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self._reaction_matrix = reaction_matrix
        self._max_num_atoms = max_num_atoms
        self._max_num_reactions = max_num_reactions
        self._virtual_length = virtual_length
        self._fpindex = fpindex
        self._init_stack_weighted_ratio = init_stack_weighted_ratio

    def __len__(self) -> int:
        return self._virtual_length

    def __getitem__(self, _: int):
        stack = create_stack(
            self._reaction_matrix,
            max_num_reactions=self._max_num_reactions,
            max_num_atoms=self._max_num_atoms,
            init_stack_weighted_ratio=self._init_stack_weighted_ratio,
        )

        data_list: list[ProjectionData] = []
        mol_seq_full = stack.mols
        mol_idx_seq_full = stack.get_mol_idx_seq()
        rxn_seq_full = stack.rxns
        rxn_idx_seq_full = stack.get_rxn_idx_seq()
        data_list.append(
            create_data(
                product=mol_seq_full[0],
                mol_seq=mol_seq_full[:1],
                mol_idx_seq=mol_idx_seq_full[:1],
                rxn_seq=rxn_seq_full[:1],
                rxn_idx_seq=rxn_idx_seq_full[:1],
                fpindex=self._fpindex,
            )
        )
        for i in range(1, len(mol_seq_full)):
            if rxn_idx_seq_full[i] is not None:
                data_list.append(
                    create_data(
                        product=mol_seq_full[i],
                        mol_seq=mol_seq_full[: i + 1],
                        mol_idx_seq=mol_idx_seq_full[: i + 1],
                        rxn_seq=rxn_seq_full[: i + 1],
                        rxn_idx_seq=rxn_idx_seq_full[: i + 1],
                        fpindex=self._fpindex,
                    )
                )

        return data_list


class ProjectionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        batch_size: int,
        num_workers: int = 4,
        max_datapoints_per_sample: int | None = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        if max_datapoints_per_sample is not None:
            self.max_datapoints: int | None = batch_size * max_datapoints_per_sample
        else:
            self.max_datapoints = None
        self.num_workers = num_workers
        self.dataset_options = kwargs

    def setup(self, stage: str | None = None) -> None:
        trainer = self.trainer
        if trainer is None:
            raise RuntimeError("The trainer is missing.")

        if not os.path.exists(self.config.chem.rxn_matrix):
            raise FileNotFoundError(
                f"Reaction matrix not found: {self.config.chem.rxn_matrix}. "
                "Please generate the reaction matrix before training."
            )
        if not os.path.exists(self.config.chem.fpindex):
            raise FileNotFoundError(
                f"Fingerprint index not found: {self.config.chem.fpindex}. "
                "Please generate the fingerprint index before training."
            )

        with open(self.config.chem.rxn_matrix, "rb") as f:
            rxn_matrix = pickle.load(f)

        with open(self.config.chem.fpindex, "rb") as f:
            fpindex = pickle.load(f)

        self.train_dataset = ProjectionDataset(
            reaction_matrix=rxn_matrix,
            virtual_length=self.config.train.val_freq * self.batch_size * trainer.world_size,
            fpindex=fpindex,
            **self.dataset_options,
        )
        self.val_dataset = ProjectionDataset(
            reaction_matrix=rxn_matrix,
            virtual_length=self.batch_size,
            fpindex=fpindex,
            **self.dataset_options,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=Collater(max_datapoints=self.max_datapoints),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=Collater(max_datapoints=self.max_datapoints),
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )


def draw_data(data: ProjectionData):
    im_list = [draw_text("START")]
    for m, r in zip(data["mol_seq"], data["rxn_seq"]):
        if r is not None:
            im_list.append(r.draw())
        else:
            im_list.append(m.draw())
    im_list.append(draw_text("END"))
    return make_grid(im_list)


def draw_batch(batch: ProjectionBatch):
    bsz = len(batch["mol_seq"])
    for b in range(bsz):
        im_list = [draw_text("START")]
        for m, r in zip(batch["mol_seq"][b], batch["rxn_seq"][b]):
            if r is not None:
                im_list.append(r.draw())
            else:
                im_list.append(m.draw())
        im_list.append(draw_text("END"))
        yield make_grid(im_list)
