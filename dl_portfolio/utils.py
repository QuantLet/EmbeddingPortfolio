import datetime as dt
import os

LOG_BASE_DIR = './dl_portfolio/log'


def create_log_dir(model_name, model_type):
    subdir = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(LOG_BASE_DIR, model_name, model_type, subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    return log_dir
