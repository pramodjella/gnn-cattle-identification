"""
Utility functions for the Cattle GNN Biometric Identification pipeline.
Provides config loading, logging, reproducibility, and common helpers.
"""

import os
import sys
import json
import random
import logging
import yaml
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(config=None):
    """Get the computation device."""
    if config and config.get('project', {}).get('device', 'auto') != 'auto':
        device = config['project']['device']
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def setup_logging(log_dir=None, name="cattle_gnn"):
    """Setup logging to both file and console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{name}_{timestamp}.log")
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_stats(stats, filepath):
    """Save statistics dictionary to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    

def load_stats(filepath):
    """Load statistics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(n):
    """Format large numbers with commas."""
    return f"{n:,}"


class Timer:
    """Simple timer for profiling pipeline stages."""
    
    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.elapsed = 0
        
    def __enter__(self):
        self.start_time = datetime.now()
        return self
        
    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        
    def __str__(self):
        return f"{self.name}: {self.elapsed:.2f}s"
