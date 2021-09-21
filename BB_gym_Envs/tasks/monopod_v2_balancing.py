import gym
import numpy as np
from BB_gym_Envs.tasks.monopod_base import MonopodBase
from gym_ignition.utils.typing import Action, Reward, Observation

class MonopodV2Balancing(MonopodBase):

    def __init__(self,
                 agent_rate: float,
                 *args,
                 **kwargs):
        super().__init__(agent_rate, **kwargs)

        self.vec_size = kwargs["vec_size"] if "vec_size" in kwargs else 60


    # # Rewarding for height and speed
    # def get_reward(self) -> Reward:
    #     # Get vertical boom angle and velocity.
    #     _,_,_,_,_,_, bp, dbp,_,_ = self.get_observation()

    #     # Discretize bp and dbp number spaces to adjust reward based on "bucket" from np.digitize
    #     bp_vec = np.linspace(0.05, self._bp_limit, self.vec_size)
    #     dbp_vec = np.linspace(-self._dbp_limit, self._dbp_limit, self.vec_size)

    #     bp_bin_multiplier = np.digitize(bp, bp_vec)
    #     dbp_bin_multiplier = np.digitize(abs(dbp), dbp_vec)

    #     # Get reward
    #     height_reward = 10 * bp_bin_multiplier
    #     speed_reward = (self.vec_size - dbp_bin_multiplier)
    #     return float(height_reward + speed_reward)




    # Rewarding for height and speed
    def calculate_reward(self, obs: Observation) -> Reward:
        # Get vertical boom angle and velocity.
        _,_,_,_,_,_, bp, dbp,_,_ = obs

        return bp
        # # Discretize bp and dbp number spaces to adjust reward based on "bucket" from np.digitize
        # bp_vec = np.linspace(0.05, self._bp_limit, self.vec_size)
        # dbp_vec = np.linspace(-self._dbp_limit, self._dbp_limit, self.vec_size)

        # bp_bin_multiplier = np.digitize(bp, bp_vec)
        # # Get reward
        # height_reward = bp_bin_multiplier if bp_bin_multiplier > 4 else 0
        # if bp_bin_multiplier > 6:
        #     height_reward += 5
        # return float(height_reward)

    # # # Zero reward so reward can be done in code with step number.
    # # def get_reward(self) -> Reward:
    # #     # Get vertical boom angle and velocity.
    # #     _,_,_,_,_,_, bp, dbp,_,_ = self.get_observation()
    # #     return 0.0 #float(height_reward)
