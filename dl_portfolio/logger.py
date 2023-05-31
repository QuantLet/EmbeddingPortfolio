import logging


class ColorFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    blue = "\x1b[38;5;39m"
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
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name, level="debug"):
    logger = logging.getLogger(name)
    if level == "debug":
        level = logging.DEBUG
    elif level == "info":
        level = logging.INFO
    elif level == "warning":
        level = logging.WARNING
    elif level == "error":
        level = logging.ERROR
    elif level == "notset":
        level = logging.NOTSET
    else:
        raise NotImplementedError()
    logger.setLevel(level)

    stdout_logger = logging.StreamHandler()
    stdout_logger.setFormatter(ColorFormatter())
    logger.addHandler(stdout_logger)
    logger.propagate = False

    return logger


LOGGER = get_logger("DL-Portfolio-Logger")
