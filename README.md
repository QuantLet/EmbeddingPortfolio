[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **EmbeddingPortfolio** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

Welcome to the EmbeddingPortfolio wiki!

In this repo will find two quantlets: NMFRB and TailRiskAERB. Please read the 
corresponding papers. To reproduce the analysis please refer to the 
corresponding quantlet.

You will also find the required package to reproduce the papers' results. 
This is the `dl_portfolio` package.

## Installation

- First, create a virtual environment with python3.8 with conda (or virtualenv):
```bash
conda create -n NAME_OF_ENV python=3.8
```
- Set the global variables path in `dl_portfolio/pathconfig.py`
- Install the package and requirements from `setup.py` with:
```bash
pip install . --upgrade
```

## Data

The data that support the findings of this study are available from Bloomberg and the Blockchain
Research Center (BRC). Restrictions apply to the availability of these data, which were used
under license for this study. Data are available on request from the corresponding authors with
the permission of Bloomberg and BRC.

Our results are available: https://drive.google.com/drive/folders/1oIGQeLlQi6rZ6L-dpxTj9A0TtjICTR6X?usp=drive_link

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

# Training config

## Intro

For autoencoder training modify the `config/ae_config.py` and for NMF training, `config/nmf_config.py`.
The training configuration parameters are shared for ae and nmf config, but of course nmf config has less configuration.

### Parameters



- `dataset`: name of dataset, 'dataset1' or 'dataset2'
- `show_plot`: boolean
- `save`: boolean: save the result
- `nmf_model`: path to nmf model weights
- `resample`: dict, resample method, ex:

```
resample = {
    'method': 'nbb',
    'where': ['train'],
    'block_length': 60,
    'when': 'each_epoch'
}
```

- `seed`: Optional, random seed
- `encoding_dim`: int, encoding dimension
- `batch_normalization`: boolean, perform batch normalization after encoding
- `uncorrelated_features`: boolean, use uncorrelated features constraint
- `weightage`: float, uncorrelated features constraint penalty
- `ortho_weightage`: float, orthogonality constraint penalty
- `l_name`: string, regularization (follow keras names`: 'l1')
- `l`: float, regularization penalty
- `activation`: string, activation function (follow keras names`: 'relu')
- `features_config = None
- `model_name`: string
- `scaler_func`: dict, scaler method`:
```
{
    'name'`: 'StandardScaler'
}
```
- `model_type`: string, ('ae_model')
- `learning_rate`: float, learning rate
- `epochs`: int, number of epochs
- `batch_size`: int
- `val_size`: int, Optional
- `test_size`: int, 0
- `label_param`: Optional
- `rescale`: Optional
- `activity_regularizer`: Optional
- `kernel_initializer`: tf.keras.initializers
- `kernel_regularizer`: tf.keras.regularizers, use orthogonality`:
```
WeightsOrthogonality(
    encoding_dim,
    weightage=ortho_weightage,
    axis=0,
    regularizer={'name'`: l_name, 'params'`: {l_name`: l}}
)
```
- `callback_activity_regularizer`: boolean, use callback (False)
- `kernel_constraint`: tf.keras.constraints, use `NonNegAndUnitNorm(max_value=1., axis=0)`
- `callbacks`: Dict, keras callbacks, ex`:
```
callbacks = {
    'EarlyStopping'`: {
        'monitor'`: 'val_loss',
        'min_delta'`: 1e-3,
        'mode'`: 'min',
        'patience'`: 100,
        'verbose'`: 1,
        'restore_best_weights'`: True
    }
}
```
- `data_specs`: Dict[Dict], with keys 0, 1, 2, ..., N, for each cv fold  with keys, 'start', 'val_start', 'test_start' (Optional) and 'end', ex`:

```
data_specs = {
    0`: {
        'start'`: '2016-06-30',
        'val_start'`: '2019-11-13',
        'test_start'`: '2019-12-12',
        'end'`: '2020-01-11'
    },
    1`: {
        'start'`: '2016-06-30',
        'val_start'`: '2019-12-13',
        'test_start'`: '2020-01-12',
        'end'`: '2020-02-11'
    }
}
```
