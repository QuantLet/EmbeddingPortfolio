import json, os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from dl_portfolio.logger import LOGGER
from dl_portfolio.model import build_mlp, build_mlp_with_cash_bias, EIIE_model, asset_independent_model
from dl_portfolio.utils import create_log_dir
from dl_portfolio.data import build_delayed_window, features_generator, DataLoader, SeqDataLoader, reshape_to_2d_data
from dl_portfolio.evaluate import plot_train_history
from dl_portfolio.train import train
from dl_portfolio.config import config
import logging, pickle
from dl_portfolio.benchmark import market_cap_returns
from shutil import copyfile

ANN = ['mlp', 'mlp-cash-bias']
RNN = ['lstm', 'rnn']

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

    # initialize data_loader
    # if config.model_type in ANN:
    if config.model_type in ['EIIE', 'asset_independent_model']:
        data_loader = SeqDataLoader('EIIE', config.features, start_date=config.start_date, freq=config.freq,
                                    seq_len=config.seq_len, val_size=config.val_size,
                                    preprocess_param=config.preprocess, batch_size=config.batch_size,
                                    nb_folds=config.nb_folds, no_cash=config.no_cash, cv_type=config.cv_type)
    else:
        raise NotImplementedError()
        data_loader = DataLoader(config.model_type, config.features, freq=config.freq, window=config.seq_len,
                                 preprocess_param=config.preprocess, batch_size=config.batch_size,
                                 no_cash=config.no_cash)

    # Start cross-validation loop
    LOGGER.info('Start CV loop ...')
    if config.cv_type == 'fold':
        assert config.load_model == 'CV'

    for cv in range(config.nb_folds):
        LOGGER.info(f'CV {cv}')
        # create model
        LOGGER.info('Create model')
        if config.model_type == 'mlp':
            LOGGER.info(f'Build {config.model_type} model')
            model = build_mlp(input_dim=(data_loader.n_features * data_loader.n_pairs * config.seq_len),
                              layers=config.layers, output_dim=data_loader.n_assets, dropout=config.dropout)
        elif config.model_type == 'mlp-cash-bias':
            LOGGER.info(f'Build {config.model_type} model')
            model = build_mlp_with_cash_bias(input_dim=(data_loader.n_features * data_loader.n_pairs * config.seq_len),
                                             output_dim=data_loader.n_assets,
                                             batch_size=config.batch_size,
                                             n_hidden=config.n_hidden,
                                             dropout=config.dropout)
        elif config.model_type == 'EIIE':
            LOGGER.info(f'Build {config.model_type} model')
            model = EIIE_model(input_dim=(data_loader.n_pairs, config.seq_len, data_loader.n_features),
                               output_dim=data_loader.n_assets,
                               layers=config.layers,
                               dropout=config.dropout)
        elif config.model_type == 'asset_independent_model':
            LOGGER.info(f'Build {config.model_type} model')
            if config.layers[0]['type'] in ['lstm', 'conv1d']:
                input_dim = (config.seq_len, data_loader.n_features)
            else:
                input_dim = (data_loader.n_features)
            model = asset_independent_model(input_dim, output_dim=data_loader.n_assets, n_assets=data_loader.n_pairs,
                                            layers=config.layers, dropout=config.dropout)
        else:
            raise NotImplementedError()

        if config.load_model == 'CV':
            if cv > 0:
                path = os.path.join(log_dir, f'model_{cv - 1}.ckpt')
                LOGGER.info(f'Loading pretrained model from {path}')
                model.load_weights(path)

        LOGGER.info(model.summary())

        train_examples, test_examples = data_loader.get_cv_data(cv)
        # Input
        if config.model_type not in ['EIIE', 'asset_independent_model']:
            train_examples = data_loader.input_data.values[train_indices]
            test_examples = data_loader.input_data.values[test_indices]

        # Returns
        train_returns = data_loader.train_returns
        test_returns = data_loader.test_returns

        LOGGER.info(f'Train data shape: {train_examples.shape}')
        LOGGER.info(f'Train returns shape: {train_returns.shape}')
        LOGGER.info(f'Test data shape: {test_examples.shape}')
        LOGGER.info(f'Test returns shape: {test_returns.shape}')

        if 'returns' in [f['name'] for f in config.features]:
            for i in range(train_examples.shape[0]):
                assert (np.sum(train_examples[i, 1:, -1, -1] != train_returns.values[:-1, i])) == 0
                print(True)

        # Training pipeline
        LOGGER.info('Create tf.data.Dataset')
        # Train
        if config.model_type in ['EIIE', 'asset_independent_model']:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (np.transpose(train_examples, (1, 2, 3, 0)), train_returns))
            test_dataset = tf.data.Dataset.from_tensor_slices((np.transpose(test_examples, (1, 2, 3, 0)), test_returns))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_returns))
            test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_returns))
        train_dataset = train_dataset.batch(config.batch_size, drop_remainder=False)
        test_dataset = test_dataset.batch(config.batch_size, drop_remainder=False)

        # Training loop
        LOGGER.info('Start training loop ...')
        model, train_history, test_history = train(train_dataset, test_dataset, model, config.model_type,
                                                   config.loss_config['name'],
                                                   config.optimizer, config.lr_scheduler, config.n_epochs,
                                                   data_loader.assets, config.trading_fee, config.log_every,
                                                   config.plot_every, no_cash=config.no_cash, clip_value=None,
                                                   train_returns=train_returns, **config.loss_config['params'])

        # plot final history and save
        if config.save:
            plot_train_history(train_history, test_history, save_dir=os.path.join(log_dir, f'history_{cv}.png'),
                               show=True)
        else:
            plot_train_history(train_history, test_history, show=True)

        if config.save:
            # save model
            model.save_weights(os.path.join(log_dir, f'model_{cv}.ckpt'))

        # Inference on train
        train_predictions = model.predict(features_generator(train_dataset, model_type=config.model_type),
                                          verbose=1)  # , steps=train_nb_batch)
        train_predictions = pd.DataFrame(train_predictions, columns=data_loader.assets, index=train_returns.index)

        # Inference on test
        test_predictions = model.predict(features_generator(test_dataset, model_type=config.model_type),
                                         verbose=1)  # , steps=test_nb_batch)
        test_predictions = pd.DataFrame(test_predictions, columns=data_loader.assets, index=test_returns.index)

        if config.save:
            pickle.dump(test_predictions, open(os.path.join(log_dir, f'test_prediction_{cv}.p'), 'wb'))

        # Final prediction
        fig, axs = plt.subplots(1, 2, figsize=(15, 3))
        axs[0].plot(train_predictions)
        axs[0].legend(data_loader.assets)
        axs[0].set_title('Train prediction')
        axs[1].plot(test_predictions)
        axs[1].legend(data_loader.assets)
        axs[1].set_title('Test prediction')
        if config.save:
            plt.savefig(os.path.join(log_dir, f'prediction_{cv}.png'))
        plt.show()

        # Final perf
        fig, axs = plt.subplots(1, 2, figsize=(15, 3))
        strat_perf = (train_returns * train_predictions).sum(1)
        train_returns = pd.DataFrame(train_returns, index=train_predictions.index,
                                     columns=train_predictions.columns)
        benchmark_returns, benchmark_value = market_cap_returns(train_returns, config.freq)
        axs[0].plot(benchmark_value, label='benchmark')
        axs[0].plot((strat_perf + 1).cumprod(), label='strategy')
        axs[0].legend()
        axs[0].set_title('Final train performance')

        strat_perf = (test_returns * test_predictions).sum(1)
        test_returns = pd.DataFrame(test_returns, index=test_predictions.index,
                                    columns=train_predictions.columns)
        benchmark_returns, benchmark_value = market_cap_returns(test_returns, config.freq)
        axs[1].plot(benchmark_value, label='benchmark')
        axs[1].plot((strat_perf + 1).cumprod(), label='strategy')
        axs[1].legend()
        axs[1].set_title('Final test performance')
        if config.save:
            plt.savefig(os.path.join(log_dir, f'performance_{cv}.png'))
        plt.show()
