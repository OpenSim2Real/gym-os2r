import warnings
from gym_bb.config.config import SettingsConfig
from .monopod_base import MonopodBase


class MonopodBuilder(MonopodBase):
    """
    Default parameters are used in the init method for torque,
    position/rotation limits, and reset angle. Any of these values can be
    redefined by passing in the corresponding kwargs.
    """

    def __init__(self, agent_rate, **kwargs):
        self.supported_task_modes = ['free_hip',
                                     'fixed_hip', 'fixed_hip_and_boom_yaw']
        required_kwargs = ['supported_models',
                           'task_mode', 'reward_class']
        for rkwarg in required_kwargs:
            if rkwarg not in list(kwargs.keys()):
                raise RuntimeError('Missing required kwarg: ' + rkwarg
                                   + '. We require the following kwargs, '
                                   + str(required_kwargs)
                                   + '\n in the MonopodBuilder class.'
                                   ' (These can be specified in env init)')
        if len(required_kwargs) != len(list(kwargs.keys())):
            warnings.warn('# WARNING: Supplied Kwargs, ' + str(kwargs)
                          + ' Contains more entries than expected. '
                          'Could be caused by config object.',
                          SyntaxWarning, stacklevel=2)
        self.__dict__.update(kwargs)
        try:
            cfg = kwargs['config']
        except KeyError:
            cfg = SettingsConfig()

        if self.task_mode not in self.supported_task_modes:
            raise RuntimeError(
                'task mode ' + self.task_mode + ' not supported in '
                'monopod environment.')
        try:
            xpath = 'models/' + self.task_mode + '/spaces'
            self.spaces_definition = cfg.get_config(xpath)
        except KeyError:
            raise RuntimeError(
                'task mode ' + self.task_mode + ' does not contain spaces '
                'definition in monopod environment config file.')
        super().__init__(agent_rate, **kwargs)
