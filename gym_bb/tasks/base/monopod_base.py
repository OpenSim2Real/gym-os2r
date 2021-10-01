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
    Provides the base implementation for the monopod ignition gym environment. This
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
                 spaces_definition: dict(),
                 **kwargs):

        # Initialize the Task base class
        task.Task.__init__(self, agent_rate=agent_rate)
        # Name of the monopod model
        self.model_name = None
        # Space for resetting the task
        self.reset_space = None
        self.reset_boom = 0.3
        self.spaces_definition = spaces_definition
        # Set max torque for the monopod joints
        self.spaces_definition['action'] = {
            'upper_leg_joint': [-1, 1],
            'lower_leg_joint': [-1, 1]
        }

        # Optionally overwrite the above using **kwargs
        self.__dict__.update(kwargs)

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:

        # Get names joints
        self.action_names = [*self.spaces_definition['action']]
        self.joint_names = [*self.spaces_definition['observation']]

        # Create the action space
        action_lims = np.array(list(self.spaces_definition['action'].values()))
        observation_lims = np.array(
            list(self.spaces_definition['observation'].values()))

        # Configure action limits
        high = np.array(action_lims[:, 0])
        low = np.array(action_lims[:, 1])
        action_space = gym.spaces.Box(low=low, high=high)

        # Configure reset limits
        high = np.array(observation_lims[:, [0, 2]])
        low = np.array(observation_lims[:, [1, 3]])
        # Configure the reset space - this is used to check if it exists inside
        # the reset space when deciding whether to reset.
        self.reset_space = gym.spaces.Box(low=low, high=high)

        # Configure the observation space
        observation_space = gym.spaces.Box(low=low*1.2, high=high*1.2)

        return action_space, observation_space

    def set_action(self, action: Action) -> None:
        if not self.action_space.contains(action):
            raise RuntimeError(
                "Action Space does not contain the provided action")

        # Set the force value
        model = self.world.get_model(self.model_name)

        for joint, value in zip(self.action_names, action.tolist()):
            # Set torque to value given in action
            if not model.get_joint(joint).set_generalized_force_target(value):
                raise RuntimeError(
                    "Failed to set the torque in the " + joint)

    def get_observation(self) -> Observation:

        # Get the model
        model = self.world.get_model(self.model_name)

        # Get the new joint positions and velocities
        pos = model.joint_positions(self.joint_names)
        vel = model.joint_velocities(self.joint_names)

        # Create the observation
        observation = Observation(np.array([*pos, *vel]))
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

        # Control the monopod in force mode
        for joint in self.action_names:
            upper = model.get_joint(joint)
            ok = upper.set_control_mode(scenario.JointControlMode_force)
            if not ok:
                raise RuntimeError(
                    "Failed to change the control mode of the Monopod")

        # Create a new monopod state
        pos_reset = vel_reset = [0]*len(self.joint_names)
        pos_reset[self.joint_names.index('hip_joint')] = self.reset_boom

        g_model = model.to_gazebo()
        ok_pos = g_model.reset_joint_positions(pos_reset, self.joint_names)
        ok_vel = g_model.reset_joint_velocities(vel_reset, self.joint_names)

        if not (ok_pos and ok_vel):
            raise RuntimeError("Failed to reset the monopod state")

    def get_reward(self) -> Reward:
        """
        Returns the reward Calculated in calculate_reward
        """
        obs = self.get_observation()
        return self.calculate_reward(obs)

    def calculate_reward(self, obs: Observation) -> Reward:
        """
        Calculates the reward given observation.
        Implementation left to the user
        """

        raise NotImplementedError()

    def get_state_info(self, obs: Observation) -> Tuple[Reward, bool]:
        """
        Returns the reward and is_done given a state you provide.
        """
        reward = self.calculate_reward(obs)
        done = not self.reset_space.contains(obs)
        return reward, done
