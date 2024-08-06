import logging.config

logger = logging.getLogger('Logger')


def setup_logging():
    global logger
    logging.addLevelName(5, 'TRACE')

    def trace(self, message, *args, **kws):
        if self.isEnabledFor(5):
            self._log(5, message, args, **kws, stacklevel=2)

    logging.Logger.trace = trace

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {},
        "formatters": {
            "informative": {
                "format": '%(asctime)s - %(levelname)s - %(message)s - Line: %(lineno)d - File: %(filename)s - Path: ' +
                          '%(pathname)s',
                "datefmt": "%m/%d/%Y %I:%M:%S %p"
            },
            "simple": {
                "format": '%(asctime)s - %(levelname)s - %(message)s - Line: %(lineno)d',
                "datefmt": "%I:%M"
            }

        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "TRACE",
                "formatter": "informative",
                "filename": "logs.log",
                "maxBytes": 1_000_000,
                "backupCount": 3
            }
        },
        "loggers": {
            "root": {"level": "TRACE", "handlers": ["stdout", "file"]},
            "ale_py.roms": {"level": "INFO"}
        }
    }
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger('Logger')


def main() -> None:
    logger.trace("This is the trace message")
    logger.debug("This is the debug message")
    logger.info("This is the info message")
    logger.warning("This is the warning message")
    logger.error("This is the error message")
    logger.critical("This is the critical message")


if __name__ == "__main__":
    setup_logging()
    main()
