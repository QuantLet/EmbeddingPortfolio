# dl-portfolio

## Installation

- First, create a virtual environment with python3.8 with conda (or virtualenv):
```bash
conda create -n NAME_OF_ENV python=3.8
```
- Install the requirements from `setup.py` with:
```bash
pip install . --upgrade
```

## Data

The data is available here: https://drive.google.com/drive/folders/1HEou7cOtJDuAnXwWnZVuiLpwJYsc-cQ-?usp=sharing

- Download the data folder at the root. It contains the price data: `data/dataset1/dataset1.csv` and 
`data/dataset1/dataset2.csv`.
- To reproduce the result you need to additionally download the folders:
  - `final_models` where you will find the 
  autoencoder and convex NMF models as well as the Markowitz and Robust Markowitz portfolio allocation.
  - `activationProba` where you will find the input and output from ARMA-GARCH modelling for the activations
- Finally, our portfolio weights and corresponding evaluation used for the paper are saved in the `performance` folder.

## AE and NMF training

The training is done using main.py. Please check the arguments in the file.
The configuration for training are located in `dl_portolfio/config/`
- `nmf_config.py` for NMF training
- `ae_config.py` for AE training

### Run NMF on dataset 1

The configuration for running NMF training and experiments are in `dl_portolfio/config/nmf_config.py`
- Copy `nmf_config_dataset1.py` in dl_portolfio/config in `nmf_config.py`
- Then run 
```bash
python main.py --n=N_EXPERIMENT --n_jobs=N_PARALLEL_JOBS --run=nmf
```

### Run NMF on dataset 2

- Copy `nmf_config_dataset2.py` in dl_portolfio/config in `nmf_config.py`
- Then run 
```bash
python main.py --n=N_EXPERIMENT --n_jobs=N_PARALLEL_JOBS --run=nmf
```

### Run AE on dataset 1

The configuration for running AE training and experiments are in `dl_portolfio/config/ae_config.py`
- Copy `ae_config_dataset1.py` in dl_portolfio/config in `ae_config.py`
- Then run 
```bash
python main.py --n=N_EXPERIMENT --n_jobs=N_PARALLEL_JOBS --run=ae
```

### Run AE on dataset 2

The configuration for running AE training and experiments are in `dl_portolfio/config/ae_config.py`
- Copy `ae_config_dataset2.py` in dl_portolfio/config in `ae_config.py`
- Then run 
```bash
python main.py --n=N_EXPERIMENT --n_jobs=N_PARALLEL_JOBS --run=ae
```
## ARMA-GARCH modelling

ARMA-GARCH modelling is done using R in `activationProba`.
- First prepare the data using `create_lin_activation.py` and specify the base_dir where you saved the AE result
```bash
python create_lin_activation.py --base_dir=final_models/ae/dataset1/m_0_dataset1_nbb_resample_bl_60_seed_0_1647953383912806
```
- repeat for dataset2

Before running the script, define the parameters in `config/config.json`

- Dataset1: Copy config_dataset1.json in config.json and run activationProba.R
- Dataset2: Copy config_dataset2.json in config.json and run activationProba.R

## Prediction and backtest result

- For AE and NMF model, this is done using `performance.py` script. Check the script argument.
- For ARMA-GARCH model, this is done using `hedge_performance.py` script after running `performance.py`. 
Check the script argument. You also need to modify the paths for your outputs of garch and ae modelling directly in
the code: DATA_BASE_DIR_1, GARCH_BASE_DIR_1, PERF_DIR_1, DATA_BASE_DIR_2, GARCH_BASE_DIR_2, PERF_DIR_2.

## Analysis

Finally, you can look at the notebooks which produce all figure in the paper. Modify the output folders at the beginning of the files with your own outputs.

- Backtest.ipynb: Analysis of the backtest result and production of various statistics
- Interpretation.ipynb: Interpretation of the embedding
- activationProba.ipynb: Analysis of the hedged strategies