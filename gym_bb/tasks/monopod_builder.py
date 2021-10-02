import numpy as np
from .monopod_base import MonopodBase
from scenario import core as scenario
from .rewards import calculate_balancing_rewards
from gym_ignition.utils.typing import Reward, Observation


class MonopodBuilder(MonopodBase):
    """
    Default parameters are used in the init method for torque, position/rotation
    limits, and reset angle. Any of these values can be redefined by passing in
    the corresponding kwargs.
    """

    def __init__(self, agent_rate, **kwargs):

        self.spaces_definition = {}
        env_setup = {
            # Free hip
            'monopod_v1': self._create_original,
            # Fixed hip
            'monopod_v1_fh': self._create_fh,
            # Fixed Hip and boom yaw
            'monopod_v1_fh_fby': self._create_fh_fby
        }
        print(kwargs)
        try:
            env_setup[kwargs['supported_models'][0]]()
        except KeyError:
            raise RuntimeError(
                'Model ' + kwargs['supported_models'][0] + 'not supported in'
                'monopod environment')

        super().__init__(agent_rate, **kwargs)

    def calculate_reward(self, obs: Observation) -> Reward:
        reward_setup = {
            'Standing_v1': calculate_balancing_rewards.standing_v1,
        }
        return reward_setup[self.reward_calculation_type](obs)

    def reset_task(self) -> None:

        if self.model_name not in self.world.model_names():
            raise RuntimeError("Monopod model not found in the world")

        # Get the model
        model = self.world.get_model(self.model_name)

        # Control the monopod in force mode
        for joint in self.action_names:
            upper = model.get_joint(joint)
            ok = upper.set_control_mode(scenario.JointControlMode_force)
            if not ok:
                raise RuntimeError(
                    "Failed to change the control mode of the Monopod")

        # Create a new monopod state
        pos_reset = vel_reset = [0]*len(self.joint_names)
        pos_reset[self.joint_names.index(
            'planarizer_02_joint')] = self.reset_boom

        g_model = model.to_gazebo()
        ok_pos = g_model.reset_joint_positions(pos_reset, self.joint_names)
        ok_vel = g_model.reset_joint_velocities(vel_reset, self.joint_names)

        if not (ok_pos and ok_vel):
            raise RuntimeError("Failed to reset the monopod state")

    def _create_original(self):
        self.spaces_definition['observation'] = {
            # pos high, pos low, vel high, vel low
            'upper_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'lower_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            "planarizer_02_joint": [1.57, -1.57, np.inf, -np.inf],
            "planarizer_01_joint": [np.inf, -np.inf, np.inf, -np.inf],
            'hip_joint': [3.14, -3.14, np.inf, -np.inf]
            }

    def _create_fh(self):
        self.spaces_definition['observation'] = {
            # pos high, pos low, vel high, vel low
            'upper_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'lower_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            "planarizer_02_joint": [1.57, -1.57, np.inf, -np.inf],
            "planarizer_01_joint": [np.inf, -np.inf, np.inf, -np.inf]
            }

    def _create_fh_fby(self):
        self.spaces_definition['observation'] = {
            # pos high, pos low, vel high, vel low
            'upper_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            'lower_leg_joint': [3.14, -3.14, np.inf, -np.inf],
            "planarizer_02_joint": [1.57, -1.57, np.inf, -np.inf]
            }
