import logging
from loguru import logger
from rich.logging import RichHandler

import warnings

warnings.filterwarnings(
    "ignore", message="Valid config keys have changed in V2:.*", category=UserWarning
)


class LiteLLMFilter(logging.Filter):
    def filter(self, record):
        return not (record.name == "LiteLLM" and record.levelno == logging.INFO)


# Remove the default loguru handler
logger.remove()

# Add a new handler using RichHandler for console output
logger.add(
    RichHandler(markup=True, show_time=False),  # Enable rich markup for colored output
    level="INFO",  # Set the logging level
    format="{message}",
    backtrace=True,  # Include the backtrace in the log
    diagnose=True,  # Include diagnostic information in the log
)

# Add another handler for saving debug logs to a file
logger.add(
    "debug_logs.log",  # File path for the log file
    level="DEBUG",  # Set the logging level to DEBUG
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",  # Log format
    rotation="5 MB",  # Rotate the log file when it reaches 10 MB
    retention=2,  # Keep a maximum of 2 log files
    backtrace=True,  # Include the backtrace in the log
    diagnose=True,  # Include diagnostic information in the log
)
