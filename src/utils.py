import os
import yaml
from datetime import datetime

def load_config():
    """
    Load project configuration from config.yaml
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def create_folder(path):
    """
    Create folder if it does not exist
    """
    os.makedirs(path, exist_ok=True)

def get_timestamp():
    """
    Return current timestamp as string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def log(message):
    """
    Simple logging function with timestamp
    """
    print(f"[{get_timestamp()}] {message}")


if __name__ == "__main__":
    log("Testing utils.py functions...")

    cfg = load_config()
    print("Config loaded:", cfg)

    test_folder = os.path.join(os.path.dirname(__file__), 'test_folder')
    create_folder(test_folder)
    log(f"Test folder created at {test_folder}")
