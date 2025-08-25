from typing import Dict, Any

def recursive_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """
    Recursively merges two dictionaries.
    Modifies target in place.
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            recursive_merge(target[key], value)
        else:
            target[key] = value