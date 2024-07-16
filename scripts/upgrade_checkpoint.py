import click
import torch
from omegaconf import OmegaConf

from chemprojector.models.old.projector_wrapper import ProjectorWrapper as WrapperV1
from chemprojector.models.wrapper import ChemProjectorWrapper as WrapperV2

prefix_mapping: dict[str, str] = {
    "model.atom_emb.": "model.encoder.atom_emb.",
    "model.bond_emb.": "model.encoder.bond_emb.",
    "model.enc.": "model.encoder.enc.",
    "model.in_token.": "model.decoder.in_token.",
    "model.in_reaction.": "model.decoder.in_reaction.",
    "model.in_fingerprint.": "model.decoder.in_fingerprint.",
    "model.pe.": "model.decoder.pe_dec.",
    "model.dec.": "model.decoder.dec.",
    "model.out_token.": "model.token_head.mlp.",
    "model.out_reaction.": "model.reaction_head.mlp.",
    "model.out_fingerprint.": "model.fingerprint_head.mlp.",
}


@click.command()
@click.option("--v2-cfg", type=OmegaConf.load, required=True)
@click.option("--v1-ckpt", type=click.Path(exists=True), required=True)
@click.option("--v2-ckpt", type=click.Path(exists=False), required=True)
def main(v2_cfg, v1_ckpt, v2_ckpt):
    assert v2_cfg.version == 2, "New config should have version attribute with value 2"

    print("Building models...")
    model_v1 = WrapperV1.load_from_checkpoint(v1_ckpt)
    model_v2 = WrapperV2(v2_cfg)

    print("Converting state dict...")
    state_dict_v1 = model_v1.state_dict()
    state_dict_upgraded: dict[str, torch.Tensor] = {}
    for k, v in state_dict_v1.items():
        k_new: str | None = None
        for prefix, new_prefix in prefix_mapping.items():
            if k.startswith(prefix):
                if k_new is None:
                    k_new = k.replace(prefix, new_prefix)
                else:
                    raise ValueError(f"Key {k} matched multiple prefixes")
        if k_new is None:
            raise ValueError(f"Key {k} did not match any prefixes")

        print(f"- {k} -> {k_new}")
        state_dict_upgraded[k_new] = v

    result = model_v2.load_state_dict(state_dict_upgraded)
    print("State dict validation result:", result)

    print(f"Saving upgraded checkpoint to {v2_ckpt}...")
    torch.save(
        {
            "hyper_parameters": model_v2.hparams,
            "state_dict": model_v2.state_dict(),
        },
        v2_ckpt,
    )


if __name__ == "__main__":
    main()
