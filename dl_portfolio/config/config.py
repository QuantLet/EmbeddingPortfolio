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
    100: 0.001,
    140: 0.0001
}

momentum = 0.8
n_epochs = 100
batch_size = 64
log_every = 1
plot_every = 20
test_size = 300

# Strategy, loss function
freq=3600
no_cash = False
benchmark = 0.
annual_period = 1
trading_fee = 0.0075

# Output
save = True