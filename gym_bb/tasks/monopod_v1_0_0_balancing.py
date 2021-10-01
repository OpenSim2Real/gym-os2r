import numpy as np
from .base.monopod_builder import MonopodBuilder
from gym_ignition.utils.typing import Reward, Observation


class MonopodBalancing(MonopodBuilder):

    def __init__(self,
                 agent_rate: float,
                 reward_balance_position: bool = True,
                 **kwargs):

        super().__init__(agent_rate, **kwargs)

    def calculate_reward(self, obs: Observation) -> Reward:

        # Calculate the reward
        done = not self.reset_space.contains(obs)
        reward = 1.0 if not done else 0.0
        if self._reward_balance_position:
            def gaussian(x, mu, sig):
                return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-((x - mu)/sig)*((x - mu)/sig)/2)
            # Get the observation
            u, _, l, _, _, _, bp, dbp, _, dby = obs
            # Guassian function distribution of reward around the desired angle of the boom.
            # The variance is determined by the current speed. More speed = more variance
            mu = self.reset_boom
            sig = 75 * abs(dbp / self._dbp_limit)
            alpha = 1
            reward = alpha * gaussian(bp, mu, sig)
        return reward
