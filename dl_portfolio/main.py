import json, os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from dl_portfolio.logger import LOGGER
from dl_portfolio.model import build_etf_mlp, build_etf_mlp_with_cash_bias
from dl_portfolio.utils import create_log_dir
from dl_portfolio.data import build_delayed_window, features_generator, DataLoader
from dl_portfolio.evaluate import plot_train_history
from dl_portfolio.train import train
from dl_portfolio.config import config
import logging, pickle
from shutil import copyfile

US_10Y_BOND = 0.0093

if __name__ == '__main__':
    if config.save:
        log_dir = create_log_dir(config.model_name, config.model_type)
        LOGGER.info(f'Create log dir: {log_dir}')
        LOGGER.addHandler(logging.FileHandler(os.path.join(log_dir, 'logs.log')))
        copyfile('./dl_portfolio/config/config.py',
                 os.path.join(log_dir, 'config.py'))

    if config.seed is None:
        seed = np.random.randint(0, 100)
        LOGGER.info(f'Set random seed {seed}')
        np.random.seed(seed)
    else:
        LOGGER.info('Set seed')
        seed = config.seed
        np.random.seed(seed)

    features = [{'name': 'close'}, {'name': 'returns', 'params': {'time_period': 1}}]

    # initialize data_loader
    data_loader = DataLoader(features, freq=config.freq)

    # create model
    LOGGER.info('Create model')
    if config.model_type == 'mlp':
        LOGGER.info(f'Build {config.model_type} model')
        model = build_etf_mlp(input_dim=(data_loader.n_features * data_loader.n_pairs * config.seq_len),
                              output_dim=data_loader.n_assets,
                              n_hidden=config.n_hidden,
                              dropout=config.dropout)
    elif config.model_type == 'mlp-cash-bias':
        LOGGER.info(f'Build {config.model_type} model')
        model = build_etf_mlp_with_cash_bias(input_dim=(data_loader.n_features * data_loader.n_pairs * config.seq_len),
                                             output_dim=data_loader.n_assets,
                                             batch_size=config.batch_size,
                                             n_hidden=config.n_hidden,
                                             dropout=config.dropout)
    else:
        raise NotImplementedError()

    if config.load_model:
        path = os.path.join(log_dir, config.load_model)
        LOGGER.info(f'Loading pretrained model from {path}')
        model.load_weights(path)

    LOGGER.info(model.summary())

    # Start cross-validation loop
    LOGGER.info('Start CV loop ...')
    for cv in data_loader.cv_indices:
        LOGGER.info(f'CV {cv}')
        LOGGER.info('Train / test split')
        train_indices = data_loader.cv_indices[cv]['train']
        test_indices = data_loader.cv_indices[cv]['test']
        LOGGER.info(f'Nb train examples: {len(train_indices)}')
        LOGGER.info(f'Nb test examples: {len(test_indices)}')
        train_returns = data_loader.returns[train_indices]
        train_examples = data_loader.input_data[train_indices]
        test_returns = data_loader.returns[test_indices]
        test_examples = data_loader.input_data[test_indices]

        LOGGER.info('Preprocessing ...')
        scaler = preprocessing.MinMaxScaler([-1, 1])
        LOGGER.info('Fit to train set and transform')
        scaler.fit(train_examples)
        train_examples = scaler.transform(train_examples)
        LOGGER.info('Transform test set')
        test_examples = scaler.transform(test_examples)

        LOGGER.info('Build delayed input sequence and corresponding returns')
        # Train set
        train_examples = build_delayed_window(train_examples, seq_len=config.seq_len, return_3d=False)
        train_returns = build_delayed_window(train_returns, seq_len=config.seq_len, return_3d=True)
        # Returns correspond to last value of sequence
        train_returns = train_returns[:, -1, :]

        # Test set
        test_examples = build_delayed_window(test_examples, seq_len=config.seq_len, return_3d=False)
        test_returns = build_delayed_window(test_returns, seq_len=config.seq_len, return_3d=True)
        # Train returns correspond to last value of sequence
        test_returns = test_returns[:, -1, :]

        LOGGER.info(f'Train data shape: {train_examples.shape}')
        LOGGER.info(f'Train returns shape: {train_returns.shape}')
        LOGGER.info(f'Test data shape: {test_examples.shape}')
        LOGGER.info(f'Test returns shape: {test_returns.shape}')

        # Training pipeline
        LOGGER.info('Create tf.data.Dataset')
        # Train
        nb_batch = len(train_examples) // config.batch_size
        drop_first = np.remainder(len(train_examples), config.batch_size)
        train_examples = train_examples[drop_first:, :]
        n_features = train_examples.shape[-1]
        train_returns = train_returns[drop_first:, :]

        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_returns))
        train_dataset = train_dataset.batch(config.batch_size, drop_remainder=False)
        # Test
        test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_returns))
        test_dataset = test_dataset.batch(config.batch_size, drop_remainder=config.model_type == 'mlp-cash-bias')

        # Training loop
        LOGGER.info('Start training loop ...')
        model, train_history, test_history = train(train_dataset, test_dataset, model, config.lr_scheduler,
                                                   config.momentum,
                                                   config.n_epochs, data_loader.assets, config.benchmark, config.annual_period,
                                                   config.trading_fee, config.log_every, config.plot_every,
                                                   no_cash=config.no_cash,
                                                   clip_value=None, train_returns=train_returns)

        # plot final history and save
        if config.save:
            plot_train_history(train_history, test_history, save_dir=os.path.join(log_dir, 'history.png'), show=True)
        else:
            plot_train_history(train_history, test_history, show=True)

        if config.save:
            # save model
            model.save_weights(os.path.join(log_dir, 'model'))

        # Inference on train
        train_predictions = model.predict(features_generator(train_dataset), verbose=1)
        # Inference on test
        test_predictions = model.predict(features_generator(test_dataset), verbose=1)

        if config.save:
            pickle.dump(test_predictions, open(os.path.join(log_dir, 'test_prediction.p'), 'wb'))

        # Final prediction
        fig, axs = plt.subplots(1, 2, figsize=(15, 3))
        axs[0].plot(train_predictions)
        axs[0].legend(data_loader.assets)
        axs[0].set_title('Train prediction')
        axs[1].plot(test_predictions)
        axs[1].legend(data_loader.assets)
        axs[1].set_title('Test prediction')
        if config.save:
            plt.savefig(os.path.join(log_dir, 'prediction.png'))
        plt.show()

        # Final perf
        fig, axs = plt.subplots(1, 2, figsize=(15, 3))
        strat_perf = (train_returns[:len(train_predictions), :] * train_predictions).sum(1)
        eq_port = train_returns[:len(train_predictions), :].mean(1)
        axs[0].plot((eq_port + 1).cumprod(), label='equally weighted')
        axs[0].plot((strat_perf + 1).cumprod(), label='equally strategy')
        axs[0].legend()
        axs[0].set_title('Final train performance')
        strat_perf = (test_returns[:len(test_predictions), :] * test_predictions).sum(1)
        eq_port = test_returns[:len(test_predictions), :].mean(1)
        axs[1].plot((eq_port + 1).cumprod(), label='equally weighted')
        axs[1].plot((strat_perf + 1).cumprod(), label='equally strategy')
        axs[1].legend()
        axs[1].set_title('Final test performance')
        if config.save:
            plt.savefig(os.path.join(log_dir, 'performance.png'))
        plt.show()
