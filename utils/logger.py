"""
Logger utilities for training
"""

import logging
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime


class TrainingLogger:
    """Simple training logger that logs to file and console."""
    
    def __init__(self, log_dir: Path, name: str = "train"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.log_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Metrics log (JSON lines)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.metrics_list = []
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def log(self, metrics: Dict[str, Any]):
        """
        Log metrics dictionary.
        
        Args:
            metrics: Dictionary of metric_name: value
        """
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Write to file (JSON lines)
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        self.metrics_list.append(metrics)
    
    def get_metrics(self):
        """Get all logged metrics."""
        return self.metrics_list
    
    def save_summary(self):
        """Save summary of metrics."""
        if not self.metrics_list:
            return
        
        summary_file = self.log_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.metrics_list, f, indent=2)


def setup_logger(log_dir: Path, name: str = "train") -> TrainingLogger:
    """
    Setup training logger.
    
    Args:
        log_dir: Directory to save logs
        name: Logger name
        
    Returns:
        TrainingLogger instance
    """
    logger = TrainingLogger(log_dir, name)
    logger.info(f"Logger initialized: {log_dir}")
    return logger


if __name__ == "__main__":
    # Test logger
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = setup_logger(Path(tmpdir))
        
        logger.info("Testing logger")
        logger.warning("This is a warning")
        
        # Log metrics
        logger.log({'epoch': 0, 'loss': 1.5, 'lr': 0.001})
        logger.log({'epoch': 1, 'loss': 1.2, 'lr': 0.0009})
        
        logger.save_summary()
        
        print("✅ Logger tests passed!")
