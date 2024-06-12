from setuptools import setup

setup(
    name="chemprojector",
    version="0.1.0",
    packages=["chemprojector", "guacamol", "fcd"],
    package_dir={
        "chemprojector": "chemprojector",
        "guacamol": "third_party/guacamol/guacamol",
        "fcd": "third_party/FCD/fcd",
    },
)
