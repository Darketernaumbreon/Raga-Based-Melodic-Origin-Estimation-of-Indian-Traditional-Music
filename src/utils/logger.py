import logging
from pathlib import Path
import sys

def setup_logger(experiment_dir: Path) -> logging.Logger:
    """
    Sets up a logger that outputs to both the console and a file in the experiment directory.
    """
    logger = logging.getLogger("RaagAnalysis")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate logs if setup_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler
    log_file = experiment_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
