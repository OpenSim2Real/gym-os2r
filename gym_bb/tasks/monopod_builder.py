import numpy as np
import warnings
from .monopod_base import MonopodBase


class MonopodBuilder(MonopodBase):
    """
    Default parameters are used in the init method for torque,
    position/rotation limits, and reset angle. Any of these values can be
    redefined by passing in the corresponding kwargs.
    """

    def __init__(self, agent_rate, **kwargs):
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
                          + ' Contains more entries than expected.',
                          SyntaxWarning, stacklevel=2)

        self.__dict__.update(kwargs)
        self.spaces_definition = {}
        obs_space = self.obs_factory(self.task_mode)
        self.spaces_definition['observation'] = obs_space()
        super().__init__(agent_rate, **kwargs)

    def obs_factory(self, task_mode):
        if task_mode == 'free_hip':
            return self._free_hip
        elif task_mode == 'fixed_hip':
            return self._fixed_hip
        elif task_mode == 'fixed_hip_and_boom_yaw':
            return self._fixed_hip_and_boom_yaw
        else:
            raise RuntimeError(
                'task mode ' + task_mode + ' not supported in '
                'monopod environment.')

    def _free_hip(self):
        return {
            # pos high, pos low, vel high, vel low
            'upper_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'lower_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            "planarizer_02_joint": [1.57, -1.57, np.inf, -np.inf],
            "planarizer_01_joint": [np.inf, -np.inf, np.inf, -np.inf],
            'hip_joint': [3.14, -3.14, np.inf, -np.inf]
            }

    def _fixed_hip(self):
        return {
            # pos high, pos low, vel high, vel low
            'upper_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'lower_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            "planarizer_02_joint": [1.57, -1.57, np.inf, -np.inf],
            "planarizer_01_joint": [np.inf, -np.inf, np.inf, -np.inf]
            }

    def _fixed_hip_and_boom_yaw(self):
        return {
            # pos high, pos low, vel high, vel low
            'upper_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'lower_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            "planarizer_02_joint": [1.57, -1.57, np.inf, -np.inf]
            }
