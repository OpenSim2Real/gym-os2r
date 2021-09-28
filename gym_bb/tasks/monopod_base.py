import abc
import gym
import numpy as np
from typing import Tuple
from gym_ignition.base import task
from scenario import core as scenario
from gym_ignition.utils.typing import Action, Reward, Observation
from gym_ignition.utils.typing import ActionSpace, ObservationSpace
from abc import ABC, abstractmethod


class MonopodBase(task.Task, abc.ABC):

    """
    Provides the base implementation for the MonopodV1 ignition gym environment. This
    class does the following:

        1. Defines some important parameters of the robot. Sets position and rotation limits.
           Because of this, is_done is true when these are exceeded.
        2. Implements the methods for env.step(): get_reward, get_observation, is_done
        3. Provides methods to configure action/observation spaces, and perform an action

    Default parameters are used in the init method for torque, position/rotation limits, and
    reset angle. Any of these values can be redefined by passing in the corresponding kwargs.

    For the user, this class makes no assumption on the goal of the tasks so the get_reward method
    is left intentionally blank. Also, because the default limits can be changed, the is_done
    method can be easily overwritten for further flexibility.
    """

    def __init__(self,
                 agent_rate: float,
                 **kwargs):

        # Initialize the Task base class
        task.Task.__init__(self, agent_rate=agent_rate)

        # Name of the monopod model
        self.model_name = None

        # Space for resetting the task
        self.reset_space = None

        # Max torque for upper/lower leg
        self.max_torque_upper_leg = 1
        self.max_torque_lower_leg = 1

        # Boom reset angle
        self.reset_boom = 0.30

        # Variable limits
        # Need to set the max joint positions, velocities, torques
        # u = upper leg reference frame
        # l = lower leg reference frame
        # h = hip reference frame
        # bp = boom pitch joint - up and down movement
        # by = boom yaw joint - circe movement around

        # Upper leg position and speed
        self._u_limit = 3.14  # rad
        self._du_limit = np.deg2rad(10 * 360)  # rad / s
        # Lower leg position and speed
        self._l_limit = 1.57  # rad
        self._dl_limit = np.deg2rad(10 * 360)  # rad / s
        # Hip position and speed
        self._h_limit = 4  # rad
        self._dh_limit = np.deg2rad(10 * 360)  # rad / s
        # Boom up and down angle
        self._bp_limit = 1.57  # rad
        self._dbp_limit = np.deg2rad(10 * 360)  # rad / s
        # Boom circular rotation
        self._by_limit = 4  # rad - should always be true.
        self._dby_limit = np.deg2rad(10 * 360)  # rad / s

        # Optionally overwrite the above using **kwargs
        self.__dict__.update(kwargs)

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:

        # Create the action space
        action_space = gym.spaces.Box(low=np.array([-self.max_torque_upper_leg, -self.max_torque_lower_leg]),
                                      high=np.array(
                                          [self.max_torque_upper_leg,  self.max_torque_lower_leg]),
                                      dtype=np.float64)
        # Configure reset limits
        high = np.array([
            self._u_limit,
            self._du_limit,
            self._l_limit,
            self._dl_limit,
            self._h_limit,
            self._dh_limit,
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
            self._h_limit,
            self._dh_limit,
            self._bp_limit-self._bp_limit-0.05,
            self._dbp_limit,
            self._by_limit,
            self._dby_limit,
        ])

        # Configure the reset space - this is used to check if it exists inside the reset space when deciding whether to reset.
        self.reset_space = gym.spaces.Box(low=low,
                                          high=high,
                                          dtype=np.float64)

        # Configure the observation space
        obs_high = high.copy() * 1.2
        obs_low = low.copy() * 1.2
        observation_space = gym.spaces.Box(low=obs_low,
                                           high=obs_high,
                                           dtype=np.float64)

        return action_space, observation_space

    def set_action(self, action: Action) -> None:
        if not self.action_space.contains(action):
            raise RuntimeError(
                "Action Space does not contain the provided action")
        # Get the force value
        torque_upper_leg, torque_lower_leg = action.tolist()
        # Set the force value
        model = self.world.get_model(self.model_name)

        # Set torque to value given in action
        if not model.get_joint("upper_leg_joint").set_generalized_force_target(torque_upper_leg):
            raise RuntimeError(
                "Failed to set the torque in the upper leg joint")

        if not model.get_joint("lower_leg_joint").set_generalized_force_target(torque_lower_leg):
            raise RuntimeError(
                "Failed to set the torque in the lower leg joint")

    def get_observation(self) -> Observation:

        # Get the model
        model = self.world.get_model(self.model_name)

        # Get the new joint positions and velocities
        u, l, h, bp, by = model.joint_positions([
            "upper_leg_joint",
            "lower_leg_joint",
            "hip_joint",
            "planarizer_02_joint",
            "planarizer_01_joint"
            ])
        du, dl, dh, dbp, dby = model.joint_velocities([
            "upper_leg_joint",
            "lower_leg_joint",
            "hip_joint",
            "planarizer_02_joint",
            "planarizer_01_joint"
            ])

        # Create the observation
        observation = Observation(
            np.array([u, du, l, dl, h, dh, bp, dbp, by, dby]))
        # Return the observation
        return observation

    def is_done(self) -> bool:

        # Get the observation
        observation = self.get_observation()

        # The environment is done if the observation is outside its space
        done = not self.reset_space.contains(observation)

        return done

    def reset_task(self) -> None:

        if self.model_name not in self.world.model_names():
            raise RuntimeError("Monopod model not found in the world")

        # Get the model
        model = self.world.get_model(self.model_name)

        # Control the cart in force mode
        upper = model.get_joint("upper_leg_joint")
        ok_mode = upper.set_control_mode(scenario.JointControlMode_force)
        lower = model.get_joint("lower_leg_joint")
        ok_mode = ok_mode and lower.set_control_mode(
            scenario.JointControlMode_force)

        if not ok_mode:
            raise RuntimeError(
                "Failed to change the control mode of the Monopod")

        # Create a new monopod state
        #
        du, dl, dh, dbp, dby = self.np_random.uniform(
            low=-0.05, high=0.05, size=(5,))
        u, l, h, bp, by = self.np_random.uniform(
            low=-0.005, high=0.005, size=(5,))
        bp += self.reset_boom
        u = self.np_random.uniform(low=-0.6, high=0.6)
        l = -u

        ok_reset_pos = model.to_gazebo().reset_joint_positions([u, l, h, bp, by],
                                                               ["upper_leg_joint",
                                                                "lower_leg_joint",
                                                                "hip_joint",
                                                                "planarizer_02_joint",
                                                                "planarizer_01_joint"
                                                                ])
        ok_reset_vel = model.to_gazebo().reset_joint_velocities([du, dl, dh, dbp, dby],
                                                                ["upper_leg_joint",
                                                                 "lower_leg_joint",
                                                                 "hip_joint",
                                                                 "planarizer_02_joint",
                                                                 "planarizer_01_joint"
                                                                 ])

        if not (ok_reset_pos and ok_reset_vel):
            raise RuntimeError("Failed to reset the monopod state")

    def get_reward(self) -> Reward:
        """
        Returns the reward. Implementation left to the user
        """
        obs = self.get_observation()
        return self.calculate_reward(obs)

    def calculate_reward(self, obs: Observation) -> Reward:
        """
        Returns the reward. Implementation left to the user
        """

        raise NotImplementedError()

    def get_state_info(self, obs: Observation) -> Tuple[Reward, bool]:
        """
        Returns the reward and is_done given a state you provide. Implementation left to the user
        """
        reward = self.calculate_reward(obs)
        done = not self.reset_space.contains(obs)
        return reward, done
