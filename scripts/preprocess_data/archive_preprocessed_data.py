import pathlib
import subprocess
import uuid

import click

from chemprojector.chem.mol import read_mol_file
from chemprojector.tools.crypt import encrypt_message, save_encrypted_pack

_default_sdf_path = pathlib.Path("data/Enamine_Rush-Delivery_Building_Blocks-US_223244cmpd_20231001.sdf")


@click.command()
@click.option("--out-archive", type=click.Path(path_type=pathlib.Path), default="data/processed.zip")
@click.option("--out-key", type=click.Path(path_type=pathlib.Path), default="data/processed.key")
@click.option("--sdf", type=click.Path(path_type=pathlib.Path), default=_default_sdf_path)
@click.option("--seed", type=int, default=20240501)
@click.option("--num-mols", type=int, default=10)
def main(out_archive: pathlib.Path, out_key: pathlib.Path, sdf: pathlib.Path, seed: int, num_mols: int):
    if out_archive.exists():
        click.confirm(f"File exists: {out_archive}. Overwrite?", abort=True)
        out_archive.unlink()
    if out_key.exists():
        click.confirm(f"File exists: {out_key}. Overwrite?", abort=True)
        out_key.unlink()

    data_dir = pathlib.Path("data").absolute()
    processed_dir = data_dir / "processed"
    if not processed_dir.exists():
        raise FileNotFoundError(f"Directory not found: {processed_dir}")

    password = str(uuid.uuid4())
    print(f"Password: {password}")

    enc_pack, key_mols = encrypt_message(message=password, mols=read_mol_file(sdf), seed=seed, num_keys=num_mols)
    print("Key molecules:")
    for mol in key_mols:
        print(f"- {mol.csmiles}")
    save_encrypted_pack(enc_pack, out_key)

    subprocess.run(
        ["zip", "--password", password, "-r", str(out_archive.absolute()), "processed/"],
        cwd=data_dir,
        check=True,
        stdin=subprocess.PIPE,
        capture_output=True,
    )


if __name__ == "__main__":
    main()
