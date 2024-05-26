from setuptools import setup

setup(
    name="chemprojector",
    version="0.1.0",
    packages=["chemprojector", "guacamol", "fcd"],
    package_dir={
        "chemprojector": "./",
        "guacamol": "third_party/guacamol",
        "fcd": "third_party/FCD",
    },
)
