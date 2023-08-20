import logging # 日志记录用

def logger_obj(logger_name, level=logging.DEBUG, verbose=0):
    # 就是一个日志记录器.可以放到utils里
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ("%(asctime)s - %(levelname)s - %(funcName)s (%(lineno)d):  %(message)s")
    datefmt = '%Y-%m-%d %I:%M:%S %p'
    log_format = logging.Formatter(format_string, datefmt)

    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    if verbose == 1:
        # Creating and adding the console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    return logger