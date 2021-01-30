- `seq_len`, type=int, default=5: Input sequence length
- `model_type`, type=str, default="mlp"
- `model_name`, type=str, default="etf"
- `n_epochs`, type=int, default=150
- `batch_size`, type=int, default=64 
- `log_every`, type=int, default=1: Epoch logs frequency
- `plot_every`, type=int, default=2: Plots frequency
- `no_cash`, type=boolean, default=False: Implement a portfolio without cash
- `n_hidden`, type=int, default=3: Number of hidden layers
- `dropout`, type=float, default=None, 0: Dropout rate to apply after each hidden layer
- `learning_rate`, type=float, default=0.001: Learning rate
- `lr_scheduler = {
    0: 0.01,
    5: 0.001,
    40: 0.0001
}`, type=dict: Learning rate scheduler
- `momentum`,type=float, default=0.8: Momentum parameter in SGD
- `seed`, type=int, default=None: Seed for reproducibility, if not set use default
- `test_size`, type=int, default=300: Test size
- `benchmark`, type=float, default=0.: Benchmark for excess Sharpe Ratio
- `annual_period`, type=float, default=252: Period to annualize sharpe ratio
- `trading_fee`, type=float, default=0.0001: Trading fee
- `load_model`, type=str, default=None: Model checkpoint path
- `save`, type=boolean, default=False: Save outputs