import numpy as np
from .monopod_base import MonopodBase


class MonopodBuilder(MonopodBase):
    def __init__(self, agent_rate, kwargs):

        build = {
            'monopod_v1': self._create_original(agent_rate, kwargs),
            'monopod_v1_fh': self._create_fh(agent_rate, kwargs),
            'monopod_v1_fh_fby': self._create_fh_fby(agent_rate, kwargs),
        }
        try:
            build[self.simp_model_names]
        except KeyError:
            raise RuntimeError(
                'Simp model ' + self.simp_model_names + 'not supported.')

    def _create_original(self,
                         agent_rate: float,
                         reward_balance_position: bool = True,
                         **kwargs):
        self.spaces_definition = {}
        self.spaces_definition['observation'] = {
            # pos high, pos low, vel high, vel low
            'upper_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'lower_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'hip_joint': [3.14, -3.14, np.inf, -np.inf],
            "planarizer_02_joint": [3.14, -3.14, np.inf, -np.inf],
            "planarizer_01_joint": [3.14, -3.14, np.inf, -np.inf]
            }

        super().__init__(agent_rate, self.spaces_definition, **kwargs)

    def _create_fh(self,
                   agent_rate: float,
                   reward_balance_position: bool = True,
                   **kwargs):
        self.spaces_definition = {}
        self.spaces_definition['observation'] = {
            # pos high, pos low, vel high, vel low
            'upper_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'lower_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            "planarizer_02_joint": [3.14, -3.14, np.inf, -np.inf],
            "planarizer_01_joint": [3.14, -3.14, np.inf, -np.inf]
            }

        super().__init__(agent_rate, self.spaces_definition, **kwargs)

    def _create_fh_fby(self,
                       agent_rate: float,
                       reward_balance_position: bool = True,
                       **kwargs):
        self.spaces_definition = {}
        self.spaces_definition['observation'] = {
            # pos high, pos low, vel high, vel low
            'upper_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'lower_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            "planarizer_02_joint": [3.14, -3.14, np.inf, -np.inf]
            }

        super().__init__(agent_rate, self.spaces_definition, **kwargs)
