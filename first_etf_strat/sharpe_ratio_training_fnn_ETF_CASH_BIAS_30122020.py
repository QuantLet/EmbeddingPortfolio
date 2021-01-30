import json, os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from utils.config import LOGGER
from first_etf_strat.model import build_etf_mlp, build_etf_mlp_with_cash_bias
from utils.utils import create_log_dir
from first_etf_strat.data import build_delayed_window, features_generator
from first_etf_strat.evaluate import plot_train_history
from first_etf_strat.train import train

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("ETF FNN")
    parser.add_argument("--seq-len", type=int, default=5, help="Input sequence length")
    parser.add_argument("--model-type", type=str, default="mlp")
    parser.add_argument("--model-name", type=str, default="etf")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=1, help="Epoch logs frequency")
    parser.add_argument("--plot-every", type=int, default=20, help="Plots frequency")
    parser.add_argument("--no-cash", action='store_true', help="Implement a portfolio without cash")
    parser.add_argument("--n-hidden", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate to apply after each hidden layer")
    # parser.add_argument("--long-short", action='store_true', help="Implement a long-short portfolio stategy")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum parameter in SGD")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility, if not set use default")
    parser.add_argument("--test-size", type=int, default=300, help="Test size")
    parser.add_argument("--benchmark", type=float, default=0., help="Risk free rate for excess Sharpe Ratio")
    parser.add_argument("--annual-period", type=float, default=0., help="Period to annualize sharpe ratio")
    parser.add_argument("--trading-fee", type=float, default=0.0001, help="Trading fee")
    parser.add_argument("--load-model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--save", action="store_true", help="Save outputs")
    args = parser.parse_args()

    US_10Y_BOND = 0.0093

    if args.seed is None:
        seed = np.random.randint(0, 100)
        LOGGER.info(f'Set random seed {seed}')
        np.random.seed(seed)
    else:
        LOGGER.info('Set seed')
        seed = args.seed
        np.random.seed(seed)

    if args.save:
        log_dir = create_log_dir(args.model_name, args.model_type)
        LOGGER.info(f'Create log dir: {log_dir}')
        config = vars(args)
        config['random_seed'] = seed
        json.dump(config, open(os.path.join(log_dir, 'config.json'), 'w'))

    LOGGER.info('Load data')
    dfdata = pd.read_pickle('./first_etf_strat/data/clean_VTI_AGG_DBC.p')
    assets = np.unique(dfdata.columns.get_level_values(0)).tolist()
    dates = dfdata.index
    if not args.no_cash:
        assets = assets + ['cash']
    n_assets = len(assets)
    LOGGER.info(dfdata.head())

    plt.plot(dfdata.loc[:, pd.IndexSlice[:, 'close']].droplevel(1, 1))
    plt.show()

    LOGGER.info('Create input and returns array ...')
    # get input data
    data = dfdata.loc[:, pd.IndexSlice[:, 'close']].droplevel(0, 1).values
    # drop last price since returns and features should be shifted by one
    data = data[:-1, :]
    # get returns for next period
    returns = dfdata.loc[:, pd.IndexSlice[:, 'returns']].droplevel(0, 1).values  # returns at time t
    # drop first row since it is nan
    returns = returns[1:, :]  # returns at time t+1
    # add cash in returns
    returns = np.concatenate([returns, np.zeros(len(returns)).reshape(-1, 1)], 1)
    # daily_risk_free_rate = (1 + US_10Y_BOND) ** (1 / 3650) - 1
    # returns[:, -1] = daily_risk_free_rate

    LOGGER.info('Preprocessing ...')
    # Preprocessing
    LOGGER.info('Train / test split')
    train_returns = returns[:-args.test_size].astype(np.float32)
    train_examples = data[:-args.test_size].astype(np.float32)
    test_returns = returns[-args.test_size:].astype(np.float32)
    test_examples = data[-args.test_size:].astype(np.float32)

    LOGGER.info('MinMaxScaler with range [-1,1]')
    scaler = preprocessing.MinMaxScaler([-1, 1])
    LOGGER.info('Fit to train set and transform')
    scaler.fit(train_examples)
    train_examples = scaler.transform(train_examples)
    LOGGER.info('Transform test set')
    test_examples = scaler.transform(test_examples)

    for i in range(train_examples.shape[-1]):
        plt.plot(train_examples[:, i])
        plt.show()

    LOGGER.info('Build delayed input sequence and corresponding returns')
    # Train set
    train_examples = build_delayed_window(train_examples, seq_len=args.seq_len, return_3d=False)
    train_returns = build_delayed_window(train_returns, seq_len=args.seq_len, return_3d=True)
    # Returns correspond to last value of sequence
    train_returns = train_returns[:, -1, :]

    # Test set
    test_examples = build_delayed_window(test_examples, seq_len=args.seq_len, return_3d=False)
    test_returns = build_delayed_window(test_returns, seq_len=args.seq_len, return_3d=True)
    # Train returns correspond to last value of sequence
    test_returns = test_returns[:, -1, :]

    LOGGER.info(f'Train data shape: {train_examples.shape}')
    LOGGER.info(f'Train returns shape: {train_returns.shape}')
    LOGGER.info(f'Test data shape: {test_examples.shape}')
    LOGGER.info(f'Test returns shape: {test_returns.shape}')

    # Traning pipeline
    LOGGER.info('Create tf.data.Dataset')
    # Train
    nb_batch = len(train_examples) // args.batch_size
    drop_first = np.remainder(len(train_examples), args.batch_size)
    train_examples = train_examples[drop_first:, :]
    n_features = train_examples.shape[-1]
    train_returns = train_returns[drop_first:, :]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_returns))
    train_dataset = train_dataset.batch(args.batch_size, drop_remainder=False)
    # Test
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_returns))
    test_dataset = test_dataset.batch(args.batch_size, drop_remainder=args.model_type == 'mlp-cash-bias')

    # model
    LOGGER.info('Create model')
    if args.model_type == 'mlp':
        LOGGER.info(f'Build {args.model_type} model')
        model = build_etf_mlp(input_dim=(n_features),
                              output_dim=n_assets,
                              n_hidden=args.n_hidden,
                              dropout=args.dropout)
    elif args.model_type == 'mlp-cash-bias':
        LOGGER.info(f'Build {args.model_type} model')
        model = build_etf_mlp_with_cash_bias(input_dim=(n_features),
                                             output_dim=n_assets,
                                             batch_size=args.batch_size,
                                             n_hidden=args.n_hidden,
                                             dropout=args.dropout)
    else:
        raise NotImplementedError()

    if args.load_model:
        path = os.path.join(log_dir, args.load_model)
        LOGGER.info(f'Loading pretrained model from {path}')
        model.load_weights(path)

    LOGGER.info(model.summary())
    # Training loop
    LOGGER.info('Start training ...')
    # for debugging
    # tf.debugging.enable_check_numerics()

    model, train_history, test_history = train(train_dataset, test_dataset, model, args.learning_rate, args.momentum,
                                               args.n_epochs, assets, args.benchmark, args.annual_period,
                                               args.trading_fee, args.log_every, args.plot_every, no_cash=args.no_cash,
                                               clip_value=None, train_returns=train_returns)

    # plot final history and save
    if args.save:
        plot_train_history(train_history, test_history, save_dir=os.path.join(log_dir, 'history.png'), show=True)
    else:
        plot_train_history(train_history, test_history, show=True)

    if args.save:
        # save model
        model.save_weights(os.path.join(log_dir, 'model'))

    # Inference on train
    train_predictions = model.predict(features_generator(train_dataset), verbose=1)
    # Inference on test
    test_predictions = model.predict(features_generator(test_dataset), verbose=1)

    # Final prediction
    fig, axs = plt.subplots(1, 2, figsize=(15, 3))
    axs[0].plot(train_predictions)
    axs[0].legend(assets)
    axs[0].set_title('Train prediction')
    axs[1].plot(test_predictions)
    axs[1].legend(assets)
    axs[1].set_title('Test prediction')
    if args.save:
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
    if args.save:
        plt.savefig(os.path.join(log_dir, 'performance.png'))
    plt.show()
