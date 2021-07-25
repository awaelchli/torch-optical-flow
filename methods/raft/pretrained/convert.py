import torch
from typing import Dict, Any


def strip_module(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """ Strips the 'module.' prefix from the state-dict keys of a model checkpoint. """
    mapping = {old_key: old_key.replace("module.", "") for old_key in state_dict.keys()}
    for old_key, new_key in mapping.items():
        state_dict[new_key] = state_dict[old_key]
        del state_dict[old_key]
    return state_dict
