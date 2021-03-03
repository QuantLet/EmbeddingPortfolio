import json, os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dl_portfolio.logger import LOGGER
from dl_portfolio.metrics import np_portfolio_returns
from dl_portfolio.model import build_mlp, build_mlp_with_cash_bias, EIIE_model, asset_independent_model, stacked_asset_model
from dl_portfolio.utils import create_log_dir, get_best_model_from_dir
from dl_portfolio.data import build_delayed_window, features_generator, DataLoader, SeqDataLoader, reshape_to_2d_data
from dl_portfolio.evaluate import plot_train_history
from dl_portfolio.train import train, pretrain, online_training
from dl_portfolio.config import config
import logging, pickle
from dl_portfolio.benchmark import market_cap_returns, equally_weighted_returns
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
        seed = np.random.randint(0, 10)
        LOGGER.info(f'Set random seed {seed}')
        np.random.seed(seed)
    else:
        LOGGER.info('Set seed')
        seed = config.seed
        np.random.seed(seed)

    # initialize data_loader
    # if config.model_type in ANN:
    if config.model_type in ['EIIE', 'asset_independent_model', 'stacked_asset_model']:
        data_loader = SeqDataLoader('EIIE', config.features, start_date=config.start_date, freq=config.freq,
                                    path=config.path, pairs=config.pairs,
                                    seq_len=config.seq_len, val_size=config.val_size,
                                    preprocess_param=config.preprocess, batch_size=config.batch_size,
                                    lookfront=config.lookfront,
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
        if config.save:
            cv_log_dir = os.path.join(log_dir, f'{cv}')
        else:
            cv_log_dir = None
        # create model
        LOGGER.info('Create model')
        feed_prev_weights = False
        online = False
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
            if config.layers[0]['type'] in ['lstm', 'conv1d', 'gru']:
                input_dim = (config.seq_len, data_loader.n_features)
            else:
                input_dim = (data_loader.n_features)
            if config.layers[-1]['type'] == 'softmax_with_weights':
                feed_prev_weights = True
                online = True
            assert config.no_cash
            model = asset_independent_model(input_dim, output_dim=data_loader.n_assets, n_assets=data_loader.n_pairs,
                                            layers=config.layers, dropout=config.dropout)
        elif config.model_type == 'stacked_asset_model':
            LOGGER.info(f'Build {config.model_type} model')
            if config.layers[0]['type'] in ['lstm', 'conv1d', 'gru']:
                input_dim = (config.seq_len, data_loader.n_features)
            else:
                input_dim = (data_loader.n_features)
            if config.layers[-1]['type'] == 'softmax_with_weights':
                feed_prev_weights = True
                online = True
            assert config.no_cash
            model = stacked_asset_model(input_dim, output_dim=data_loader.n_assets, n_assets=data_loader.n_pairs,
                                        layers=config.layers, dropout=config.dropout)


        else:
            raise NotImplementedError()

        if config.load_model == 'CV':
            if cv > 0:
                model_file = get_best_model_from_dir(cv_log_dir)
                path = os.path.join(cv_log_dir, model_file)
                LOGGER.info(f'Loading pretrained model from {path}')
                model.load_weights(path)

        LOGGER.info(model.summary())

        train_examples, test_examples = data_loader.get_cv_data(cv)
        # Input
        if config.model_type not in ['EIIE', 'asset_independent_model', 'stacked_asset_model']:
            train_examples = data_loader.input_data.values[train_indices]
            test_examples = data_loader.input_data.values[test_indices]

        # Returns
        train_returns = data_loader.train_returns
        test_returns = data_loader.test_returns

        LOGGER.info(f'Train data shape: {train_examples.shape}')
        LOGGER.info(f'Train returns shape: {train_returns.shape}')
        LOGGER.info(f'Test data shape: {test_examples.shape}')
        LOGGER.info(f'Test returns shape: {test_returns.shape}')

        # if 'returns' in [f['name'] for f in config.features]:
        #     for i in range(train_examples.shape[0]):
        #         if config.lookfront > 0:
        #             assert (np.sum(
        #                 train_examples[i, config.lookfront:, -1, -1] != train_returns.values[:-config.lookfront,
        #                                                                 i])) == 0
        #         else:
        #             assert (np.sum(
        #                 train_examples[i, :, -1, -1] != train_returns.values[:, i])) == 0
        #
        #         print(True)

        # Training pipeline
        LOGGER.info('Create tf.data.Dataset')
        # Train
        if config.model_type in ['EIIE', 'asset_independent_model', 'stacked_asset_model']:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (list(range(len(train_examples[0]))), np.transpose(train_examples, (1, 2, 3, 0)), train_returns))
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (list(range(len(test_examples[0]))), np.transpose(test_examples, (1, 2, 3, 0)), test_returns))
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_returns))
            test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_returns))
        train_dataset = train_dataset.batch(config.batch_size, drop_remainder=False)
        test_dataset = test_dataset.batch(config.batch_size, drop_remainder=False)

        # Training loop
        LOGGER.info('Start training loop ...')
        if online:
            model, train_history, test_history, portfolio_weights = pretrain(train_dataset, model, config.model_type,
                                                                             config.loss_config['name'],
                                                                             config.optimizer, config.lr_scheduler,
                                                                             config.n_epochs, data_loader.assets,
                                                                             config.trading_fee, config.log_every,
                                                                             config.plot_every, no_cash=config.no_cash,
                                                                             clip_value=None,
                                                                             train_returns=train_returns,
                                                                             **config.loss_config['params'],
                                                                             save=config.save, log_dir=cv_log_dir,
                                                                             feed_prev_weights=feed_prev_weights)
            rolling_train_size = config.batch_size  # int(4 * config.batch_size)
            train_examples = train_examples[:, -rolling_train_size:, :, :]
            portfolio_weights = portfolio_weights[-rolling_train_size - 1:, :]
            np_train_returns = train_returns.values[-rolling_train_size:, :]
            model, strat_return, test_prediction = online_training(train_examples, np_train_returns, test_examples,
                                                                   test_returns.values, portfolio_weights, model,
                                                                   config.model_type,
                                                                   config.loss_config['name'],
                                                                   config.optimizer, config.lr_scheduler, 5,
                                                                   32,
                                                                   data_loader.assets, config.trading_fee,
                                                                   config.log_every,
                                                                   config.plot_every, no_cash=config.no_cash,
                                                                   **config.loss_config['params'],
                                                                   save=config.save, log_dir=cv_log_dir,
                                                                   feed_prev_weights=feed_prev_weights)
            strat_return = pd.DataFrame(strat_return, index=test_returns.index,
                                        columns=['returns'])
            test_prediction = pd.DataFrame(test_prediction, index=test_returns.index,
                                           columns=test_returns.columns)
            benchmark_returns, benchmark_value = market_cap_returns(test_returns, config.freq,
                                                                    trading_fee=config.trading_fee)

            plt.figure(figsize=(20, 10))
            plt.plot(benchmark_value, label='benchmark')
            plt.plot(np.exp(np.cumsum(strat_return)), label='strategy')
            plt.legend()
            plt.show()

            plt.plot(test_prediction)
            plt.show()
        else:
            model, train_history, test_history = train(train_dataset, test_dataset, model, config.model_type,
                                                       config.loss_config['name'],
                                                       config.optimizer, config.lr_scheduler, config.n_epochs,
                                                       data_loader.assets, config.trading_fee, config.log_every,
                                                       config.plot_every, no_cash=config.no_cash, clip_value=None,
                                                       train_returns=train_returns, **config.loss_config['params'],
                                                       save=config.save, log_dir=cv_log_dir,
                                                       feed_prev_weights=feed_prev_weights)

        # plot final history and save
        if config.save:
            plot_train_history(train_history, test_history, save_dir=os.path.join(cv_log_dir, f'history_{cv}.png'),
                               show=True)
        else:
            plot_train_history(train_history, test_history, show=True)

        if config.save:
            # save model
            model.save_weights(os.path.join(cv_log_dir, f'model_final.ckpt'))

        # Inference on train
        train_predictions = model.predict(features_generator(train_dataset, model_type=config.model_type),
                                          verbose=1)  # , steps=train_nb_batch)
        train_predictions = pd.DataFrame(train_predictions, columns=data_loader.assets, index=train_returns.index)

        # Inference on test
        test_predictions = model.predict(features_generator(test_dataset, model_type=config.model_type),
                                         verbose=1)  # , steps=test_nb_batch)
        test_predictions = pd.DataFrame(test_predictions, columns=data_loader.assets, index=test_returns.index)

        if config.save:
            pickle.dump(test_predictions, open(os.path.join(cv_log_dir, f'test_prediction.p'), 'wb'))

        # Final prediction
        fig, axs = plt.subplots(1, 2, figsize=(15, 3))
        axs[0].plot(train_predictions)
        axs[0].legend(data_loader.assets)
        axs[0].set_title('Train prediction')
        axs[1].plot(test_predictions)
        axs[1].legend(data_loader.assets)
        axs[1].set_title('Test prediction')
        if config.save:
            plt.savefig(os.path.join(cv_log_dir, f'prediction.png'))
        plt.show()

        # Final perf
        fig, axs = plt.subplots(1, 2, figsize=(15, 3))
        if config.no_cash:
            initial_position = np.array([[1 / data_loader.n_assets] * data_loader.n_assets])
        else:
            initial_position = np.array(np.array([[0] * (data_loader.n_assets - 1)]))
        strat_perf_no_fee, strat_perf = np_portfolio_returns(train_predictions, train_returns,
                                                             initial_position=initial_position,
                                                             trading_fee=config.trading_fee,
                                                             cash_bias=not config.no_cash)
        train_returns = pd.DataFrame(train_returns, index=train_predictions.index,
                                     columns=train_predictions.columns)
        if config.benchmark == 'marketcap':
            benchmark_returns, benchmark_value = market_cap_returns(train_returns, config.freq)
        elif config.benchmark == 'equally_weighted':
            benchmark_returns, benchmark_value = equally_weighted_returns(train_returns)
        else:
            raise NotImplementedError(f"Benchmark '{config.benchmark}' not implemented")

        axs[0].plot(benchmark_value, label='benchmark')
        axs[0].plot(np.exp(np.cumsum(strat_perf)), label='strategy')
        axs[0].legend()
        axs[0].set_title('Final train performance')

        if config.no_cash:
            initial_position = train_predictions.iloc[-1:, :]
        else:
            initial_position = train_predictions.iloc[-1:, :-1]
        strat_perf_no_fee, strat_perf = np_portfolio_returns(test_predictions, test_returns,
                                                             initial_position=initial_position,
                                                             trading_fee=config.trading_fee,
                                                             cash_bias=not config.no_cash)
        test_returns = pd.DataFrame(test_returns, index=test_predictions.index,
                                    columns=train_predictions.columns)
        if config.benchmark == 'marketcap':
            benchmark_returns, benchmark_value = market_cap_returns(test_returns, config.freq,
                                                                    trading_fee=config.trading_fee)
        elif config.benchmark == 'equally_weighted':
            benchmark_returns, benchmark_value = equally_weighted_returns(test_returns, trading_fee=config.trading_fee)
        else:
            raise NotImplementedError(f"Benchmark '{config.benchmark}' not implemented")
        axs[1].plot(benchmark_value, label='benchmark')
        axs[1].plot()

        axs[1].plot(np.exp(np.cumsum(strat_perf)), label='strategy')
        axs[1].plot(np.exp(np.cumsum(strat_perf_no_fee)), label='strategy no fee')
        axs[1].legend()
        axs[1].set_title('Final test performance')
        if config.save:
            plt.savefig(os.path.join(cv_log_dir, f'performance.png'))
        plt.show()
        plt.plot(np.exp(np.cumsum(strat_perf - benchmark_returns)) - 1)
        plt.title('Excess return')
        if config.save:
            plt.savefig(os.path.join(cv_log_dir, f'excess_performance.png'))
        plt.show()
