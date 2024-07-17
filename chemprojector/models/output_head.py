import abc
import dataclasses
from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from chemprojector.chem.fpindex import FingerprintIndex
from chemprojector.chem.mol import Molecule


def _SimpleMLP(
    dim_in: int,
    dim_out: int,
    dim_hidden: int,
    num_layers: int = 3,
) -> Callable[[torch.Tensor], torch.Tensor]:
    num_intermediate = num_layers - 2
    layers = [
        nn.Linear(dim_in, dim_hidden),
        nn.ReLU(),
    ]
    for _ in range(num_intermediate):
        layers.append(nn.Linear(dim_hidden, dim_hidden))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dim_hidden, dim_out))
    return nn.Sequential(*layers)


class ClassifierHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dim_hidden: int | None = None):
        super().__init__()
        dim_hidden = dim_hidden or d_model * 2
        self.mlp = _SimpleMLP(d_model, num_classes, dim_hidden=dim_hidden)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)

    def get_loss(self, h: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        logits = self.predict(h)
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1)
        if mask is not None:
            mask_flat = mask.view(-1)
            total = mask_flat.sum().to(logits_flat) + 1e-6
            loss = (F.cross_entropy(logits_flat, target_flat, reduction="none") * mask_flat).sum() / total
        else:
            loss = F.cross_entropy(logits_flat, target_flat)
        return loss


_LossDict: TypeAlias = dict[str, torch.Tensor]
_AuxDict: TypeAlias = dict[str, torch.Tensor]


@dataclasses.dataclass
class ReactantRetrievalResult:
    reactants: np.ndarray
    fingerprint_predicted: np.ndarray
    fingerprint_retrieved: np.ndarray
    distance: np.ndarray
    indices: np.ndarray


class BaseFingerprintHead(nn.Module, abc.ABC):
    def __init__(self, fingerprint_dim: int):
        super().__init__()
        self._fingerprint_dim = fingerprint_dim

    @property
    def fingerprint_dim(self) -> int:
        return self._fingerprint_dim

    @abc.abstractmethod
    def predict(self, h: torch.Tensor, **kwargs) -> torch.Tensor: ...

    def retrieve_reactants(
        self,
        h: torch.Tensor,
        fpindex: FingerprintIndex,
        topk: int = 4,
        **options,
    ) -> ReactantRetrievalResult:
        """
        Args:
            h:  Tensor of shape (*batch, h_dim).
            fpindex:  FingerprintIndex
            topk:  Number of reactants to retrieve per fingerprint.
        Returns:
            - numpy Molecule array of shape (*batch, n_fps, topk).
            - numpy fingerprint array of shape (*batch, n_fps, topk, fp_dim).
        """
        fp = self.predict(h, **options)  # (*batch, n_fps, fp_dim)
        fp_dim = fp.shape[-1]
        out = np.empty(list(fp.shape[:-1]) + [topk], dtype=Molecule)
        out_fp = np.empty(list(fp.shape[:-1]) + [topk, fp_dim], dtype=np.float32)
        out_dist = np.empty(list(fp.shape[:-1]) + [topk], dtype=np.float32)
        out_idx = np.empty(list(fp.shape[:-1]) + [topk], dtype=np.int64)

        fp_flat = fp.view(-1, fp_dim)
        out_flat = out.reshape(-1, topk)
        out_fp_flat = out_fp.reshape(-1, topk, fp_dim)
        out_dist_flat = out_dist.reshape(-1, topk)
        out_idx_flat = out_idx.reshape(-1, topk)

        query_res = fpindex.query_cuda(q=fp_flat, k=topk)
        for i, q_res_subl in enumerate(query_res):
            for j, q_res in enumerate(q_res_subl):
                out_flat[i, j] = q_res.molecule
                out_fp_flat[i, j] = q_res.fingerprint
                out_dist_flat[i, j] = q_res.distance
                out_idx_flat[i, j] = q_res.index

        return ReactantRetrievalResult(
            reactants=out,
            fingerprint_predicted=fp.detach().cpu().numpy(),
            fingerprint_retrieved=out_fp,
            distance=out_dist,
            indices=out_idx,
        )

    @abc.abstractmethod
    def get_loss(
        self,
        h: torch.Tensor,
        fp_target: torch.Tensor,
        fp_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[_LossDict, _AuxDict]: ...


class MultiFingerprintHead(BaseFingerprintHead):
    def __init__(
        self,
        d_model: int,
        num_out_fingerprints: int,
        fingerprint_dim: int,
        dim_hidden: int,
        num_layers: int = 3,
        warmup_prob: float = 1.0,
    ):
        super().__init__(fingerprint_dim=fingerprint_dim)
        self.d_model = d_model
        self.num_out_fingerprints = num_out_fingerprints
        self.warmup_prob = warmup_prob
        d_out = fingerprint_dim * num_out_fingerprints
        self.mlp = _SimpleMLP(d_model, d_out, dim_hidden, num_layers=num_layers)

    def predict(self, h: torch.Tensor, **kwargs) -> torch.Tensor:
        y_fingerprint = torch.sigmoid(self.mlp(h))
        out_shape = h.shape[:-1] + (self.num_out_fingerprints, self.fingerprint_dim)
        return y_fingerprint.view(out_shape)

    def get_loss(
        self,
        h: torch.Tensor,
        fp_target: torch.Tensor,
        fp_mask: torch.Tensor,
        *,
        warmup: bool = False,
        **kwargs,
    ) -> tuple[_LossDict, _AuxDict]:
        bsz, seqlen, _ = h.shape
        y_fingerprint = self.mlp(h)  # (bsz, seqlen, n_fps * fp_dim)
        fp_shape = [bsz, seqlen, self.num_out_fingerprints, self.fingerprint_dim]
        y_fingerprint = y_fingerprint.view(fp_shape)
        fp_target = fp_target[:, :, None, :].expand(fp_shape)
        loss_fingerprint_all = F.binary_cross_entropy_with_logits(
            y_fingerprint,
            fp_target,
            reduction="none",
        ).sum(dim=-1)
        loss_fingerprint_min, fp_select = loss_fingerprint_all.min(dim=-1)
        if self.training and warmup:
            loss_fingerprint_avg = loss_fingerprint_all.mean(dim=-1)
            loss_fingerprint = torch.where(
                torch.rand_like(loss_fingerprint_min) < self.warmup_prob,
                loss_fingerprint_avg,
                loss_fingerprint_min,
            )
        else:
            loss_fingerprint = loss_fingerprint_min
        loss_fingerprint = (loss_fingerprint * fp_mask).sum() / (fp_mask.sum() + 1e-6)

        return {"fingerprint": loss_fingerprint}, {"fp_select": fp_select}
