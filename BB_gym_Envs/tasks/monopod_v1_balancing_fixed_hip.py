import gym
import numpy as np
from BB_gym_Envs.tasks.monopod_base import MonopodBase
from typing import Tuple
from scenario import core as scenario
from gym_ignition.utils.typing import Action, Reward, Observation
from gym_ignition.utils.typing import ActionSpace, ObservationSpace



class MonopodV1BalancingFixedHip(MonopodBase):

    def __init__(self,
                 agent_rate: float,
                 reward_balance_position: bool = True,
                 **kwargs):

        super().__init__(agent_rate, **kwargs)
        self._reward_balance_position = reward_balance_position

    def calculate_reward(self, obs: Observation) -> Reward:

        # Calculate the reward
        done = not self.reset_space.contains(obs)
        reward = 1.0 if not done else 0.0
        if self._reward_balance_position:
            def gaussian(x, mu, sig):
                return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-((x - mu)/sig)*((x - mu)/sig)/2)
            # Get the observation
            u,_,l,_,_,_, bp, dbp,_, dby = obs
            # Guassian function distribution of reward around the desired angle of the boom.
            # The variance is determined by the current speed. More speed = more variance
            mu = self.reset_boom
            sig = 75 * abs(dbp / self._dbp_limit)
            alpha = 1
            reward = alpha * gaussian(bp, mu, sig)
        return reward

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:

        # Create the action space
        action_space = gym.spaces.Box(low=np.array([-self.max_torque_upper_leg, -self.max_torque_lower_leg]),
                                      high=np.array([self.max_torque_upper_leg,  self.max_torque_lower_leg]),
                                      dtype=np.float32)
        # Configure reset limits
        high = np.array([
            self._u_limit,
            self._du_limit,
            self._l_limit,
            self._dl_limit,
            self._bp_limit,
            self._dbp_limit,
            self._by_limit,
            self._dby_limit,
        ])
        low = -np.array([
            self._u_limit,
            self._du_limit,
            self._l_limit,
            self._dl_limit,
            self._bp_limit-self._bp_limit-0.05,
            self._dbp_limit,
            self._by_limit,
            self._dby_limit,
        ])

        # Configure the reset space - this is used to check if it exists inside the reset space when deciding whether to reset.
        self.reset_space = gym.spaces.Box(low=low,
                                          high=high,
                                          dtype=np.float32)

        # Configure the observation space
        obs_high = high.copy() * 1.2
        obs_low = low.copy() * 1.2
        observation_space = gym.spaces.Box(low=obs_low,
                                           high=obs_high,
                                           dtype=np.float32)

        return action_space, observation_space

    def get_observation(self) -> Observation:

        # Get the model
        model = self.world.get_model(self.model_name)

        # Get the new joint positions and velocities
        u, l, bp, by = model.joint_positions([
            "upper_leg_joint",
            "lower_leg_joint",
            "planarizer_02_joint",
            "planarizer_01_joint"
            ])
        du, dl, dbp, dby = model.joint_velocities([
            "upper_leg_joint",
            "lower_leg_joint",
            "planarizer_02_joint",
            "planarizer_01_joint"
            ])

        # Create the observation
        observation = Observation(np.array([u, du, l, dl, bp, dbp, by, dby]))

        # Return the observation
        return observation
