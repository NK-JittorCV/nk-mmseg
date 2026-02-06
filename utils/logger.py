"""
Logger utility for DFormer training and testing.
Pure Python implementation without external dependencies.
"""

import os
import sys
import logging
import datetime


def setup_logger(name, log_file=None, level=logging.INFO, format_str=None):
    """Setup logger with file and console handlers."""
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class Logger:
    """Custom logger class."""
    
    def __init__(self, name, log_file=None, level=logging.INFO):
        self.logger = setup_logger(name, log_file, level)
    
    def info(self, message):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message):
        """Log debug message."""
        self.logger.debug(message)
    
    def critical(self, message):
        """Log critical message."""
        self.logger.critical(message)


class TrainingLogger:
    """Logger specifically for training progress."""
    
    def __init__(self, log_dir, name='training'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logger
        log_file = os.path.join(log_dir, f'{name}.log')
        self.logger = setup_logger(name, log_file)
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.start_time = None
    
    def start_training(self):
        """Start training timer."""
        self.start_time = datetime.datetime.now()
        self.logger.info("Training started")
    
    def log_epoch_start(self, epoch):
        """Log epoch start."""
        self.epoch = epoch
        self.logger.info(f"Epoch {epoch} started")
    
    def log_epoch_end(self, epoch, train_loss, val_loss=None, val_metrics=None):
        """Log epoch end with metrics."""
        log_msg = f"Epoch {epoch} completed - Train Loss: {train_loss:.4f}"
        
        if val_loss is not None:
            log_msg += f", Val Loss: {val_loss:.4f}"
        
        if val_metrics is not None:
            for key, value in val_metrics.items():
                log_msg += f", {key}: {value:.4f}"
        
        self.logger.info(log_msg)
    
    def log_step(self, step, loss, lr=None, metrics=None):
        """Log training step."""
        self.step = step
        log_msg = f"Step {step} - Loss: {loss:.4f}"
        
        if lr is not None:
            log_msg += f", LR: {lr:.6f}"
        
        if metrics is not None:
            for key, value in metrics.items():
                log_msg += f", {key}: {value:.4f}"
        
        self.logger.info(log_msg)
    
    def log_validation(self, epoch, metrics):
        """Log validation results."""
        log_msg = f"Validation Epoch {epoch}:"
        for key, value in metrics.items():
            log_msg += f" {key}: {value:.4f}"
        self.logger.info(log_msg)
    
    def log_best_model(self, epoch, metric_name, metric_value):
        """Log best model save."""
        self.logger.info(f"Best model saved at epoch {epoch} with {metric_name}: {metric_value:.4f}")
    
    def end_training(self):
        """End training and log total time."""
        if self.start_time:
            end_time = datetime.datetime.now()
            total_time = end_time - self.start_time
            self.logger.info(f"Training completed in {total_time}")
    
    def info(self, message):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message."""
        self.logger.error(message)


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger."""
    return setup_logger('root', log_file, log_level)


def print_log(msg, logger=None, level=logging.INFO):
    """Print message and log it."""
    if logger is None:
        print(msg)
    else:
        if level == logging.DEBUG:
            logger.debug(msg)
        elif level == logging.INFO:
            logger.info(msg)
        elif level == logging.WARNING:
            logger.warning(msg)
        elif level == logging.ERROR:
            logger.error(msg)
        elif level == logging.CRITICAL:
            logger.critical(msg)
        else:
            logger.info(msg) 