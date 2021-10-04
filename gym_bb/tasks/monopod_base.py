import abc
import gym
import numpy as np
from typing import Tuple
from gym_ignition.base import task
from gym_ignition.utils.typing import Action, Reward, Observation
from gym_ignition.utils.typing import ActionSpace, ObservationSpace
from scenario import core as scenario


class MonopodBase(task.Task, abc.ABC):

    """
    Provides the base implementation for the monopod ignition gym environment.
    This class does the following:

        1. Defines some important parameters of the robot. Sets position and
           rotation limits. Because of this, is_done is true when these are
           exceeded.
        2. Implements the methods for env.step(): get_reward, get_observation,
           is_done
        3. Provides methods to configure action/observation spaces, and perform
           an action

    For the user, this class makes no assumption on the goal of the tasks so
    the get_reward method is left intentionally blank. Also, because the
    default limits can be changed, the is_done method can be easily overwritten
    for further flexibility.
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
        # Set max torque for the monopod joints
        self.spaces_definition['action'] = {
            'upper_leg_joint': [-1, 1],
            'lower_leg_joint': [-1, 1]
        }

        # Get names joints
        self.action_names = [*self.spaces_definition['action']]
        self.joint_names = [*self.spaces_definition['observation']]

        # Create dict of index in obs for obs type
        self.observation_index = {}
        for i, joint in enumerate(self.joint_names):
            self.observation_index[joint + '_pos'] = i
            self.observation_index[joint + '_vel'] = i + len(self.joint_names)
        kwargs['observation_index'] = self.observation_index

        # Initialize Reward Class from Kwarg passed in.
        self.reward = self.reward_class(self.observation_index)
        if not self.reward.is_task_supported(self.task_mode):
            raise RuntimeError(self.task_mode
                               + ' task mode not supported by '
                               + str(self.reward) + ' reward class.')

        # Optionally overwrite the above using **kwargs
        self.__dict__.update(kwargs)

    def create_spaces(self) -> Tuple[ActionSpace, ObservationSpace]:

        # Create the action space
        action_lims = np.array(list(self.spaces_definition['action'].values()))
        observation_lims = np.array(
            list(self.spaces_definition['observation'].values()))

        # Configure action limits
        low = np.array(action_lims[:, 0])
        high = np.array(action_lims[:, 1])
        action_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)

        # Configure reset limits
        a = observation_lims
        low = np.concatenate((a[:, 1], a[:, 3]))
        high = np.concatenate((a[:, 0], a[:, 2]))
        # Configure the reset space - this is used to check if it exists inside
        # the reset space when deciding whether to reset.
        self.reset_space = gym.spaces.Box(low=low, high=high, dtype=np.float64)

        # Configure the observation space
        observation_space = gym.spaces.Box(low=low*1.2, high=high*1.2,
                                           dtype=np.float64)

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
                    "Failed to set the torque for joint: " + joint)

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
        """
        Reset_task sets the scenario backend to force controller
        For ignition simulation positions are reset in randomizer.
        """
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
        return self.reward.calculate_reward(obs)

    def get_state_info(self, obs: Observation) -> Tuple[Reward, bool]:
        """
        Returns the reward and is_done given a state you provide.
        """
        reward = self.calculate_reward(obs)
        done = not self.reset_space.contains(obs)
        return reward, done
        return reward, done
