import dataclasses
from collections.abc import Callable
from typing import TypedDict

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from chemprojector.chem.fpindex import FingerprintIndex
from chemprojector.chem.matrix import ReactantReactionMatrix
from chemprojector.chem.mol import Molecule
from chemprojector.chem.reaction import Reaction
from chemprojector.data.common import ProjectionBatch, TokenType
from chemprojector.models.transformer.graph_transformer import GraphTransformer
from chemprojector.models.transformer.positional_encoding import PositionalEncoding


@dataclasses.dataclass
class EncoderConfig:
    num_atom_classes: int = 100
    num_bond_classes: int = 10

    dim: int = 512
    depth: int = 8
    dim_head: int = 64
    edge_dim: int = 128
    heads: int = 8
    rel_pos_emb: bool = False
    output_norm: bool = False
    mlp_num_layers: int = 3


@dataclasses.dataclass
class DecoderConfig:
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    pe_max_len: int = 32
    output_norm: bool = False
    mlp_num_layers: int = 3

    fingerprint_dim: int = 256
    num_out_fingerprints: int = 1
    num_reaction_classes: int = 100


@dataclasses.dataclass
class ProjectorConfig:
    enc: EncoderConfig = dataclasses.field(default_factory=EncoderConfig)
    dec: DecoderConfig = dataclasses.field(default_factory=DecoderConfig)


class _PredictResult(TypedDict):
    y_token: torch.Tensor
    y_reaction: torch.Tensor
    y_fingerprint: torch.Tensor

    token_next: torch.Tensor

    rxn_indices_next: torch.Tensor
    reaction_next: list[list[Reaction | None]]
    rxn_scores_next: torch.Tensor

    reactant_next: list[list[Molecule | None]]
    fingerprint_next: torch.Tensor
    reactant_indices_next: torch.Tensor
    reactant_scores_next: torch.Tensor


class _GenerateResult(TypedDict):
    token_types: torch.Tensor
    rxn_indices: torch.Tensor
    reactant_fps: torch.Tensor
    reactants: list[list[Molecule | None]]
    reactions: list[list[Reaction | None]]


def SimpleMLP(dim_in, dim_out, num_layers: int = 2) -> Callable[[torch.Tensor], torch.Tensor]:
    if num_layers == 2:
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, dim_out),
        )
    elif num_layers == 3:
        return nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_out),
        )
    else:
        raise ValueError(f"num_layers must be 2 or 3, got {num_layers}")


class Projector(nn.Module):
    def __init__(self, cfg: ProjectorConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ProjectorConfig()
        self.cfg = cfg

        self.atom_emb = nn.Embedding(cfg.enc.num_atom_classes + 1, cfg.enc.dim, padding_idx=0)
        self.bond_emb = nn.Embedding(cfg.enc.num_bond_classes + 1, cfg.enc.edge_dim, padding_idx=0)
        self.enc = GraphTransformer(
            dim=cfg.enc.dim,
            depth=cfg.enc.depth,
            dim_head=cfg.enc.dim_head,
            edge_dim=cfg.enc.edge_dim,
            heads=cfg.enc.heads,
            rel_pos_emb=cfg.enc.rel_pos_emb,
            output_norm=cfg.enc.output_norm,
        )

        self.in_token = nn.Embedding(max(TokenType) + 1, cfg.dec.d_model)
        self.in_reaction = nn.Embedding(cfg.dec.num_reaction_classes, cfg.dec.d_model)
        self.in_fingerprint = SimpleMLP(cfg.dec.fingerprint_dim, cfg.dec.d_model, num_layers=cfg.dec.mlp_num_layers)
        self.pe = PositionalEncoding(
            d_model=cfg.dec.d_model,
            max_len=cfg.dec.pe_max_len,
        )
        self.dec = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=cfg.dec.d_model,
                nhead=cfg.dec.nhead,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=cfg.dec.num_layers,
            norm=nn.LayerNorm(cfg.dec.d_model) if cfg.dec.output_norm else None,
        )
        self.out_token = SimpleMLP(cfg.dec.d_model, max(TokenType) + 1, num_layers=cfg.dec.mlp_num_layers)
        self.out_reaction = SimpleMLP(cfg.dec.d_model, cfg.dec.num_reaction_classes, num_layers=cfg.dec.mlp_num_layers)
        self.out_fingerprint = SimpleMLP(
            cfg.dec.d_model,
            cfg.dec.num_out_fingerprints * cfg.dec.fingerprint_dim,
            num_layers=cfg.dec.mlp_num_layers,
        )

    def encode(
        self,
        atoms: torch.Tensor,
        bonds: torch.Tensor,
        atom_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        atom_emb = self.atom_emb(atoms)
        bond_emb = self.bond_emb(bonds)
        node, _ = self.enc(nodes=atom_emb, edges=bond_emb, mask=atom_padding_mask)
        return node

    def embed_seq(
        self,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
    ) -> torch.Tensor:
        emb_token = self.in_token(token_types)
        emb_rxn = self.in_reaction(rxn_indices)
        emb_fingerprint = self.in_fingerprint(reactant_fps)
        token_types_expand = token_types.unsqueeze(-1).expand(
            [token_types.size(0), token_types.size(1), self.cfg.dec.d_model]
        )
        emb_token = torch.where(token_types_expand == TokenType.REACTION, emb_rxn, emb_token)
        emb_token = torch.where(token_types_expand == TokenType.REACTANT, emb_fingerprint, emb_token)
        emb_token = self.pe(emb_token)
        return emb_token

    def get_loss(
        self,
        code: torch.Tensor,
        code_padding_mask: torch.Tensor,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
        token_padding_mask: torch.Tensor,
        warmup: bool = False,
    ):
        bsz = code.size(0)
        emb_token = self.embed_seq(token_types, rxn_indices, reactant_fps)
        x_in = emb_token[:, :-1]
        seqlen = x_in.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=seqlen,
            dtype=x_in.dtype,
            device=x_in.device,
        )
        tgt_key_padding_mask = torch.zeros(
            [bsz, seqlen],
            dtype=causal_mask.dtype,
            device=causal_mask.device,
        ).masked_fill_(token_padding_mask[:, :-1], -torch.finfo(causal_mask.dtype).max)
        y = self.dec(
            tgt=x_in,
            memory=code,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=code_padding_mask,
            tgt_is_causal=True,
        )

        token_types_out = token_types[:, 1:]
        rxn_indices_out = rxn_indices[:, 1:]
        reactant_fps_out = reactant_fps[:, 1:]

        y_token = self.out_token(y)
        loss_token = F.cross_entropy(y_token.transpose(1, 2), token_types_out)

        y_reaction = self.out_reaction(y)
        reaction_flag = token_types_out == TokenType.REACTION
        loss_reaction = (
            F.cross_entropy(y_reaction.transpose(1, 2), rxn_indices_out, reduction="none") * reaction_flag
        ).sum() / (reaction_flag.sum() + 1e-6)

        y_fingerprint = self.out_fingerprint(y)  # (bsz, seqlen, n_fps * fp_dim)
        fp_shape = [bsz, seqlen, self.cfg.dec.num_out_fingerprints, self.cfg.dec.fingerprint_dim]
        y_fingerprint = y_fingerprint.view(fp_shape)
        reactant_fps_out = reactant_fps_out[:, :, None, :].expand(fp_shape)
        fingerprint_flag = token_types_out == TokenType.REACTANT
        loss_fingerprint_all = F.binary_cross_entropy_with_logits(
            y_fingerprint, reactant_fps_out, reduction="none"
        ).sum(dim=-1)
        loss_fingerprint_min, fp_select = loss_fingerprint_all.min(dim=-1)
        if self.training and warmup:
            loss_fingerprint_avg = loss_fingerprint_all.mean(dim=-1)
            loss_fingerprint = torch.where(
                torch.rand_like(loss_fingerprint_min) < 0.01,
                loss_fingerprint_avg,
                loss_fingerprint_min,
            )
        else:
            loss_fingerprint = loss_fingerprint_min
        loss_fingerprint = (loss_fingerprint * fingerprint_flag).sum() / (fingerprint_flag.sum() + 1e-6)
        fp_select = fp_select.flatten()[fingerprint_flag.flatten()]

        y_dict = {
            "token": y_token,
            "reaction": y_reaction,
            "fingerprint": y_fingerprint,
            "fp_select": fp_select,
        }
        loss_dict = {
            "token": loss_token,
            "reaction": loss_reaction,
            "fingerprint": loss_fingerprint,
        }
        return y_dict, loss_dict

    def forward(self, batch: ProjectionBatch, warmup: bool = False):
        code = self.encode(batch["atoms"], batch["bonds"], batch["atom_padding_mask"])
        y_dict, loss_dict = self.get_loss(
            code=code,
            code_padding_mask=batch["atom_padding_mask"],
            token_types=batch["token_types"],
            rxn_indices=batch["rxn_indices"],
            reactant_fps=batch["reactant_fps"],
            token_padding_mask=batch["token_padding_mask"],
            warmup=warmup,
        )
        return y_dict, loss_dict

    @torch.inference_mode()
    def predict(
        self,
        code: torch.Tensor,
        code_padding_mask: torch.Tensor,
        token_types: torch.Tensor,
        rxn_indices: torch.Tensor,
        reactant_fps: torch.Tensor,
        rxn_matrix: ReactantReactionMatrix,
        fpindex: FingerprintIndex,
        topk: int = 4,
        result_device: torch.device | None = None,
    ) -> _PredictResult:
        result_device = result_device or code.device
        bsz = code.size(0)
        x = self.embed_seq(token_types, rxn_indices, reactant_fps)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=x.size(1),
            dtype=x.dtype,
            device=x.device,
        )
        y = self.dec(
            tgt=x,
            memory=code,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=code_padding_mask,
        )  # (bsz, seq_len, d_model)
        y_next = y[:, -1:]  # (bsz, 1, d_model)
        y_token = self.out_token(y_next)
        y_reaction = self.out_reaction(y_next)[..., : len(rxn_matrix.reactions)]
        y_fingerprint = self.out_fingerprint(y_next)

        token_next = torch.argmax(y_token, dim=-1)  # (bsz, 1)

        rxn_scores_next, rxn_indices_next = torch.sort(y_reaction, dim=-1, descending=True)
        rxn_scores_next = rxn_scores_next[:, 0, :topk]  # (bsz, 1, n_rxn) -> (bsz, topk)
        rxn_indices_next = rxn_indices_next[:, 0, :topk]
        reaction_next: list[list[Reaction | None]] = []
        for i in range(bsz):
            if token_next[i].item() != TokenType.REACTION:
                reaction_next.append([None] * topk)
            else:
                reaction_next.append([])
                for j in range(topk):
                    ridx = int(rxn_indices_next[i, j].item())
                    reaction_next[i].append(rxn_matrix.reactions[ridx])

        fp_query = torch.sigmoid(y_fingerprint).detach()[:, 0]  # (bsz, n_fps*fp_dim)
        fp_query = fp_query.view([bsz, self.cfg.dec.num_out_fingerprints, -1])  # (bsz, n_fps, fp_dim)
        query_res = fpindex.query_cuda(fp_query, k=topk)
        fp_next_list: list[torch.Tensor] = []
        reactant_scores_next_list: list[torch.Tensor] = []
        reactant_next: list[list[Molecule | None]] = []
        reactant_indices_next_list: list[torch.Tensor] = []
        for i, q_res_subl in enumerate(query_res):
            fp_i: list[torch.Tensor] = []
            sc_i: list[float] = []
            mo_i: list[Molecule | None] = []
            mo_idx_i: list[int] = []

            for j, q_res in enumerate(q_res_subl):
                fp_i.append(torch.tensor(q_res.fingerprint, dtype=torch.float))
                sc_i.append(1 / q_res.distance)
                if token_next[i].item() != TokenType.REACTANT:
                    mo_i.append(None)
                    mo_idx_i.append(-1)
                else:
                    mo_i.append(q_res.molecule)
                    mo_idx_i.append(q_res.index)

            fp_next_list.append(torch.stack(fp_i, dim=0))
            reactant_next.append(mo_i)
            reactant_indices_next_list.append(torch.tensor(mo_idx_i))
            reactant_scores_next_list.append(torch.tensor(sc_i))

        fp_next = torch.stack(fp_next_list, dim=0).to(y_fingerprint)  # (bsz, topk, fp_dim)
        reactant_indices_next = torch.stack(reactant_indices_next_list, dim=0).to(y_fingerprint)  # (bsz, topk)
        reactant_scores_next = torch.stack(reactant_scores_next_list, dim=0).to(y_fingerprint)  # (bsz, topk)

        return {
            "y_token": y_token.to(result_device),
            "y_reaction": y_reaction.to(result_device),
            "y_fingerprint": y_fingerprint.to(result_device),
            "token_next": token_next.to(result_device),
            "rxn_indices_next": rxn_indices_next.to(result_device),
            "reaction_next": reaction_next,
            "rxn_scores_next": rxn_scores_next.to(result_device),
            "reactant_next": reactant_next,
            "fingerprint_next": fp_next.to(result_device),
            "reactant_indices_next": reactant_indices_next.to(result_device),
            "reactant_scores_next": reactant_scores_next.to(result_device),
        }

    @torch.inference_mode()
    def generate(
        self,
        atoms: torch.Tensor,
        bonds: torch.Tensor,
        atom_padding_mask: torch.Tensor,
        rxn_matrix: ReactantReactionMatrix,
        fpindex: FingerprintIndex,
        max_len: int = 24,
    ) -> _GenerateResult:
        bsz = atoms.size(0)
        fp_dim = self.cfg.dec.fingerprint_dim

        code = self.encode(atoms, bonds, atom_padding_mask)
        code_padding_mask = atom_padding_mask

        token_types = torch.full([bsz, 1], fill_value=TokenType.START, dtype=torch.long, device=atoms.device)
        rxn_indices = torch.full([bsz, 1], fill_value=0, dtype=torch.long, device=atoms.device)
        reactant_fps = torch.zeros([bsz, 1, fp_dim], dtype=torch.float, device=atoms.device)
        reactants: list[list[Molecule | None]] = [[None] for _ in range(bsz)]
        reactions: list[list[Reaction | None]] = [[None] for _ in range(bsz)]

        for _ in tqdm(range(max_len - 1)):
            pred = self.predict(
                code=code,
                code_padding_mask=code_padding_mask,
                token_types=token_types,
                rxn_indices=rxn_indices,
                reactant_fps=reactant_fps,
                rxn_matrix=rxn_matrix,
                fpindex=fpindex,
            )

            token_types = torch.cat([token_types, pred["token_next"]], dim=1)
            rxn_indices = torch.cat([rxn_indices, pred["rxn_indices_next"][:, :1]], dim=1)
            reactant_fps = torch.cat([reactant_fps, pred["fingerprint_next"][:, :1]], dim=1)
            for i, m in enumerate(pred["reactant_next"]):
                reactants[i].append(m[0])
            for i, r in enumerate(pred["reaction_next"]):
                reactions[i].append(r[0])

        return {
            "token_types": token_types,
            "rxn_indices": rxn_indices,
            "reactant_fps": reactant_fps,
            "reactants": reactants,
            "reactions": reactions,
        }


def draw_generation_results(result: _GenerateResult):
    from PIL import Image

    from chemprojector.utils.image import draw_text, make_grid

    bsz, len = result["token_types"].size()
    im_list: list[Image.Image] = []
    for b in range(bsz):
        im: list[Image.Image] = []
        for l in range(len):
            if result["token_types"][b, l] == TokenType.START:
                im.append(draw_text("START"))
            elif result["token_types"][b, l] == TokenType.END:
                im.append(draw_text("END"))
                break
            elif result["token_types"][b, l] == TokenType.REACTION:
                rxn = result["reactions"][b][l]
                if rxn is not None:
                    im.append(rxn.draw())
            elif result["token_types"][b, l] == TokenType.REACTANT:
                reactant = result["reactants"][b][l]
                if reactant is not None:
                    im.append(reactant.draw())

        im_list.append(make_grid(im))
    return im_list
