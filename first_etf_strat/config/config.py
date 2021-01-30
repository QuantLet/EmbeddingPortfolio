# Random seed
seed = None

# Data
seq_len = 5

# Model
model_type = "mlp"
model_name = "etf"
n_hidden = 3
dropout = 0

# Training
load_model = None
learning_rate = 0.001

lr_scheduler = {
    0: 0.01,
    5: 0.001,
    40: 0.0001
}

momentum = 0.8
n_epochs = 150
batch_size = 64
log_every = 1
plot_every = 20
test_size = 300

# Strategy, loss function
no_cash = False
benchmark = 0.
annual_period = 252
trading_fee = 0.0001

# Output
save = False
