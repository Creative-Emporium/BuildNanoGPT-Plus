import torch
import torch.nn as nn


class Cache:
    def __init__(self, config):
        self.config = config
        self.key_list = []
        self.value_list = []

    def reset(self):
        """Clear the cache."""
        self.key_list.clear()
        self.value_list.clear()

    def append(self, keys, values):
        """Append new keys and values to the cache."""
        self.key_list.append(keys)
        self.value_list.append(values)

    def get_keys(self):
        """Retrieve all cached keys."""
        return torch.cat(self.key_list, dim=2) if self.key_list else None

    def get_values(self):
        """Retrieve all cached values."""
        return torch.cat(self.value_list, dim=2) if self.value_list else None
