import logging
import os
import sys

def setup_logger(name="mule_detection", log_file="project/logs/pipeline.log"):
    """
    Sets up a standardized logging configuration for the hackathon project pipeline.
    Ensures clear console tracing and persistent file logging.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(linZone)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Fix formatter placeholder typo if needed, otherwise using standard:
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to create file logger handler: {e}")

    return logger

# Singleton logger instance
logger = setup_logger()
