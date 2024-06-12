:warning: Do not manually install third-party libraries because it will cause RDKit version conflicts and break the program.

These third-party libraries are automatically installed when you install the ChemProjector package by running the following command in the root directory of this repo:
```bash
pip install -e .
```

NOTE: We do not use the official GuacaMol repo because it has an unfixed issue [here](https://github.com/BenevolentAI/guacamol/pull/32).
