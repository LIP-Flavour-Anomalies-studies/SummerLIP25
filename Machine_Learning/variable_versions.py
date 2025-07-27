"""
Module to load variables used for training according to version.
"""
import json

def load_variables(version, config_path="Machine_Learning/variable_versions.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config[str(version)]