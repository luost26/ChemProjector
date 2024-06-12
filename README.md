# ChemProjector

<img src="./assets/animate.gif" alt="cover" style="width:70%;" />

:dart: Projecting Molecules into Synthesizable Chemical Spaces (ICML 2024)

[[Paper]](https://arxiv.org/abs/2406.04628)

:construction: Work in progress ...

## Install

### Environment

```bash
conda env create -f env.yml -n chemprojector
conda activate chemprojector
```

The default CUDA version is 11.8. If you have a different version, please modify the `env.yml` file accordingly.


### Building Block Data

We provide preprocessed building block data. You can download it from [here](https://drive.google.com/file/d/1scui0RZ8oeroDAafnw4jgTi3yKtXxXpe/view?usp=drive_link) and put it in the `data` directory.

However, the data is derived from Enamine the Enamine's building block catalog, which are **available only upon request**.
Therefore, you should first request the data from Enamine [here](https://enamine.net/building-blocks/building-blocks-catalog), and then run the following command to unarchive the preprocessed data. The script will check whether you have a copy of the Enamine's catalog and do the rest for you.
```bash
python unarchive_wizard.py
```

You may also process the building block data by yourself. Please refer to the `scripts/preprocess_data` directory for more details.


### Trained Weights

You can download the trained weights from [here](https://drive.google.com/drive/folders/1T9f9MsEAsAjPV8GR0pXimHKCvq97SIzs?usp=drive_link) and put them in the `data/trained_weights` directory.


## Usage

### Project Molecules

You can create a list of molecules in CSV format (example: `data/example.csv`) and run the following command to project them into the synthesizable chemical space.
```bash
python sample.py \
    --input data/example.csv \
    --output results/example.csv \
```

### Model Evaluation

:construction: Work in progress ...


## Reference

```bibtex
@inproceedings{luo2024chemprojector,
  title={Projecting Molecules into Synthesizable Chemical Spaces},
  author={Shitong Luo and Wenhao Gao and Zuofan Wu and Jian Peng and Connor W. Coley and Jianzhu Ma},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```