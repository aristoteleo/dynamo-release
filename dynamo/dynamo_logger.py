import functools
import logging
from contextlib import contextmanager


def silence_logger(name):
    """Given a logger name, silence it completely.

    :param name: name of the logger
    :type name: str
    """
    package_logger = logging.getLogger(name)
    package_logger.setLevel(logging.CRITICAL + 100)
    package_logger.propagate = False


def format_logging_message(msg, logging_level, indent_level=1, indent_space_num=4):
    indent_str = "-" * indent_space_num
    prefix = "|" + indent_str * indent_level
    if logging_level == logging.INFO:
        prefix += ">"
    elif logging_level == logging.WARNING:
        prefix += "?"
    elif logging_level == logging.CRITICAL:
        prefix += "!!"
    new_msg = prefix + " " + msg
    return new_msg


class Logger:
    """Dynamo-specific logger that sets up logging for the entire package."""

    FORMAT = "[%(asctime)s] %(levelname)7s %(message)s"

    def __init__(self, name, level=None):
        self.namespace = "main"
        self.logger = logging.getLogger(name)
        self.ch = logging.StreamHandler()
        self.ch.setFormatter(logging.Formatter(self.FORMAT))
        self.logger.addHandler(self.ch)
        self.logger.propagate = False

        # Other global initialization
        silence_logger("anndata")
        silence_logger("h5py")
        silence_logger("numba")
        silence_logger("pysam")
        silence_logger("pystan")

        if not (level is None):
            self.logger.setLevel(level)
        else:
            self.logger.setLevel(logging.INFO)

    def namespaced(self, namespace):
        """Function decorator to set the logging namespace for the duration of
        the function.

        :param namespace: the namespace
        :type namespace: str
        """

        def wrapper(func):
            @functools.wraps(func)
            def inner(*args, **kwargs):
                previous = self.namespace
                try:
                    self.namespace = namespace
                    return func(*args, **kwargs)
                finally:
                    self.namespace = previous

            return inner

        return wrapper

    @contextmanager
    def namespaced_context(self, namespace):
        """Context manager to set the logging namespace.

        :param namespace: the namespace
        :type namespace: str
        """
        previous = self.namespace
        self.namespace = namespace
        yield
        self.namespace = previous

    def namespace_message(self, message):
        """Add namespace information at the beginning of the logging message.

        :param message: the logging message
        :type message: str

        :return: namespaced message
        :rtype: string
        """
        return f"[{self.namespace}] {message}"

    def setLevel(self, *args, **kwargs):
        return self.logger.setLevel(*args, **kwargs)

    def debug(self, message, indent_level=1, *args, **kwargs):
        message = format_logging_message(message, logging.DEBUG, indent_level=indent_level)
        return self.logger.debug(self.namespace_message(message), *args, **kwargs)

    def info(self, message, indent_level=1, *args, **kwargs):
        message = format_logging_message(message, logging.INFO, indent_level=indent_level)
        return self.logger.info(self.namespace_message(message), *args, **kwargs)

    def warning(self, message, indent_level=1, *args, **kwargs):
        message = format_logging_message(message, logging.WARNING, indent_level=indent_level)
        return self.logger.warning(self.namespace_message(message), *args, **kwargs)

    def exception(self, message, indent_level=1, *args, **kwargs):
        message = format_logging_message(message, logging.ERROR, indent_level=indent_level)
        return self.logger.exception(self.namespace_message(message), *args, **kwargs)

    def critical(self, message, indent_level=1, *args, **kwargs):
        message = format_logging_message(message, logging.CRITICAL, indent_level=indent_level)
        return self.logger.critical(self.namespace_message(message), *args, **kwargs)

    def error(self, message, indent_level=1, *args, **kwargs):
        message = format_logging_message(message, logging.ERROR, indent_level=indent_level)
        return self.logger.error(self.namespace_message(message), *args, **kwargs)


class LoggerManager:

    main_logger = Logger("Dynamo")

    @staticmethod
    def get_main_logger():
        return LoggerManager.main_logger

    @staticmethod
    def get_logger(name):
        return Logger(name)
