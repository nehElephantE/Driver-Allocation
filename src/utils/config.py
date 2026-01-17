"""
Configuration management
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

class Config:
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    # Load configuration from YAML file
    def _load_config(self):
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up project paths
        self.project_root = Path(__file__).parent.parent.parent
        self._setup_paths()
        
        # Initialize logging
        self._setup_logging()
    
    def _setup_paths(self):
        # Data paths
        self.data_paths = {
            'raw': {
                'booking': self.project_root / self.config['data']['raw']['booking'],
                'participant': self.project_root / self.config['data']['raw']['participant'],
                'test': self.project_root / self.config['data']['raw']['test']
            },
            'processed': {
                'train': self.project_root / self.config['data']['processed']['train'],
                'test': self.project_root / self.config['data']['processed']['test'],
                'features': self.project_root / self.config['data']['processed']['features']
            },
            'interim': self.project_root / self.config['data']['interim']
        }
        
        # Artifact paths
        self.artifact_paths = {
            'models': self.project_root / self.config['output']['models_dir'],
            'features': self.project_root / self.config['output']['features_dir'],
            'metrics': self.project_root / self.config['output']['metrics_dir'],
            'predictions': self.project_root / self.config['output']['predictions_dir']
        }
        
        # Log path
        self.log_path = self.project_root / self.config['logging']['file']
        
        # Create directories
        for path_dict in [self.data_paths, self.artifact_paths]:
            for category, paths in path_dict.items():
                if isinstance(paths, dict):
                    for path in paths.values():
                        path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    paths.mkdir(parents=True, exist_ok=True)
        
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup logging configuration
    def _setup_logging(self):
        logging_config = self.config['logging']
        
        logging.basicConfig(
            level=getattr(logging, logging_config['level']),
            format=logging_config['format'],
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    # Get configuration value by dot notation
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    # Update configuration value by dot notation
    def update(self, key: str, value: Any):
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        
        config[keys[-1]] = value

# Singleton instance
config = Config()