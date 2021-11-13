import datetime as dt
import os
from typing import Dict
from dl_portfolio.regularizers import WeightsOrthogonality

LOG_BASE_DIR = './dl_portfolio/log'


def create_log_dir(model_name, model_type):
    subdir = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(LOG_BASE_DIR, model_name, model_type, subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    return log_dir


def get_best_model_from_dir(dir_):
    files = os.listdir(dir_)
    files = list(filter(lambda x: 'model' in x, files))
    files = [[f, f.split('e_')[-1].split('.')[0]] for f in files]
    files.sort(key=lambda x: x[1])
    file = files[-1][0]
    return file


def config_setter(config, params: Dict):
    for k in params:
        if k == 'encoding_dim':
            config.encoding_dim = params[k]
        elif k == 'ortho_weightage':
            config.ortho_weightage = params[k]
            config.kernel_regularizer = WeightsOrthogonality(
                config.encoding_dim,
                weightage=config.ortho_weightage,
                axis=0,
                regularizer={
                    'name': config.l_name,
                    'params': {config.l_name: config.l}
                }
            )
        elif k == 'weightage':
            config.weightage = params[k]
        else:
            raise NotImplementedError()

    return config
