import pathlib
import subprocess
import uuid

import click

from chemprojector.chem.mol import read_mol_file
from chemprojector.tools.crypt import encrypt_message, save_encrypted_pack

_default_sdf_path = pathlib.Path("data/Enamine_Rush-Delivery_Building_Blocks-US_223244cmpd_20231001.sdf")


@click.command()
@click.option("--message", type=str, required=True, help="The message to encrypt.")
@click.option("--out-key", type=click.Path(path_type=pathlib.Path), default="data/processed.key")
@click.option("--sdf", type=click.Path(path_type=pathlib.Path), default=_default_sdf_path)
@click.option("--seed", type=int, default=20240501)
@click.option("--num-mols", type=int, default=5)
def main(message: str, out_key: pathlib.Path, sdf: pathlib.Path, seed: int, num_mols: int):
    if out_key.exists():
        click.confirm(f"File exists: {out_key}. Overwrite?", abort=True)
        out_key.unlink()

    enc_pack, key_mols = encrypt_message(
        message=message,
        mols=[m for m, _ in zip(read_mol_file(sdf), range(10000))],
        seed=seed,
        num_keys=num_mols,
    )
    print("Key molecules:")
    for mol in key_mols:
        print(f"- {mol.csmiles}")
    save_encrypted_pack(enc_pack, out_key)


if __name__ == "__main__":
    main()
