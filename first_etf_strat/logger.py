import logging


def get_logger(name_logger):
    logger_to_ret = logging.getLogger(name_logger)

    logger_to_ret.setLevel(logging.DEBUG)

    stdout_logger = logging.StreamHandler()
    stdout_logger.setFormatter(
        logging.Formatter(
            '[%(name)s:%(filename)s:%(lineno)d] - [%(process)d] - %(asctime)s - %(levelname)s - %(message)s'
        )
    )

    logger_to_ret.addHandler(stdout_logger)
    logger_to_ret.propagate = False

    return logger_to_ret


LOGGER = get_logger("First-Etf-Strat-Logger")
