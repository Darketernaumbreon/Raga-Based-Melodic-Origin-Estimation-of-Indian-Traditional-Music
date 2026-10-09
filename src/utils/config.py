"""
Configuration Loader and Saver.

Handles loading training hyperparameters and dataset paths from YAML files under configs/.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file into a dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}

def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration dictionary to a YAML file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
