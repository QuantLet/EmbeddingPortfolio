import logging


class ColorFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    blue = '\x1b[38;5;39m'
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "[%(name)s:%(filename)s:%(lineno)d] - [%(process)d] - %(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name_logger, level="debug"):
    logger_to_ret = logging.getLogger(name_logger)
    if level == "debug":
        level = logging.DEBUG
    elif level == "info":
        level = logging.INFO
    elif level == "warning":
        level = logging.WARNING
    else:
        raise NotImplementedError()
    logger_to_ret.setLevel(level)

    stdout_logger = logging.StreamHandler()
    stdout_logger.setLevel(level)
    stdout_logger.setFormatter(ColorFormatter())
    logger_to_ret.addHandler(stdout_logger)
    logger_to_ret.propagate = False

    return logger_to_ret


LOGGER = get_logger("DL-Portfolio-Logger")
