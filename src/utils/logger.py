"""
Custom logger setup
"""
import logging
import sys
from pathlib import Path
from src.utils.config import config

class Logger:
    
    def __init__(self, name: str = None):
        self.name = name or __name__
        self.logger = logging.getLogger(self.name)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # File handler (detailed)
        file_handler = logging.FileHandler(config.log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler (simple)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Set logger level
        self.logger.setLevel(config.get('logging.level', 'INFO'))
    
    # Get the logger instance
    def get_logger(self):
        return self.logger
    
    # Log info message
    def info(self, message: str):
        self.logger.info(message)
    
    # Log debug message
    def debug(self, message: str):
        self.logger.debug(message)
    
    #Log warning message
    def warning(self, message: str):
        self.logger.warning(message)
    
    # Log error message
    def error(self, message: str):
        self.logger.error(message)
    
    # Log critical message
    def critical(self, message: str):
        self.logger.critical(message)
    
    #Log exception with traceback
    def exception(self, message: str):
        self.logger.exception(message)

# Create default logger
logger = Logger(__name__).get_logger()