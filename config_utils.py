import yaml
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


class ConfigBranch(dict):
    def __init__(self, **kwarg):
        self.__dict__ = self
        self.assign(**kwarg)
            
    
    def __call__(self, **kwarg):
        self.assign(**kwarg)
        return self.__dict__


    def __bool__(self):
        return any(self.__dict__.values())


    def assign(self, **kwarg):
        for name, attr in kwarg.items():
            if type(attr) == dict:
                attr = ConfigBranch(**attr)
            setattr(self, name, attr)
        return kwarg


def load_yaml_config(config_path):
    try:
        Loader = yaml.CLoader
    except:
        Loader = yaml.Loader

    with open(config_path) as yamlfile:
        config = yaml.load(yamlfile, Loader=Loader)
    config = ConfigBranch(**config)

    return config