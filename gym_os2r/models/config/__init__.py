from operator import getitem
from functools import reduce
import yaml
import os
from copy import copy


class BaseConfig():
    """
    Default config loads in the default configuration file. Allows the user to
    update the configuration as necessary.
    """

    def __init__(self, yaml_path: str):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, yaml_path)
        with open(filename) as f:
            self.config_dict = yaml.load(f, Loader=yaml.FullLoader)

    def set_config(self, value, xpath: str):
        """
        Sets value in the dictionary using xpath like formating. for example,
        'fixed_hip/spaces/observation/' will access the fixed_hip observatin
        space limits. This function is currently limited. It does no error
        handling or type enforcement. Furthermore, it does not support * regex
        in the xpath.
        """
        mapList = xpath.strip('/').split('/')
        reduce(getitem, mapList[:-1], self.config_dict)[mapList[-1]] = value

    def get_config(self, xpath: str):
        """
        gets value in the dictionary using xpath like formating. for example,
        'fixed_hip/spaces/observation/' will access the fixed_hip observation
        space limits. This function is currently limited since it does not
        support * regex in the xpath.
        """
        mapList = xpath.strip('/').split('/')
        return copy(reduce(getitem, mapList[:-1], self.config_dict)[mapList[-1]])


class SettingsConfig(BaseConfig):
    """
    Config object for all environment settings. Loads default settings natively
    allows user to update the values (be careful... currently the user is given
    a lot of power here)
    """

    def __init__(self, yaml_path='./default/settings.yaml'):
        super().__init__(yaml_path=yaml_path)


class RandomizerConfig(BaseConfig):
    """
    Config object for randomizer configuration file. Gives access in a way that
    is usable for the randomizer.
    """

    def __init__(self, yaml_path='./default/randomizations.yaml'):
        super().__init__(yaml_path=yaml_path)

    def get_randomizations():
        pass
