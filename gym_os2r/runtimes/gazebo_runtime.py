from typing import Optional

import gym_ignition_models
from gym_ignition import base
import gym_ignition.runtimes
from gym_ignition.utils import logger
from gym_ignition.utils.typing import *

from scenario import gazebo as scenario


class GazeboRuntime(gym_ignition.runtimes.gazebo_runtime.GazeboRuntime):
    """
    Implementation of :py:class:`~gym_ignition.base.runtime.Runtime` for the Ignition
    Gazebo simulator.

    Args:
        task_cls: The class of the handled task.
        agent_rate: The rate at which the environment is called.
        physics_rate: The rate of the physics engine.
        real_time_factor: The desired RTF of the simulation.
        physics_engine: *(optional)* The physics engine to use.
        world: *(optional)* The path to an SDF world file. The world should not contain
            any physics plugin.

    Note:
        Physics randomization is still experimental and it could change in the future.
        Physics is loaded only once, when the simulator starts. In order to change the
        physics, a new simulator should be created.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        task_cls: type,
        agent_rate: float,
        physics_rate: float,
        real_time_factor: float,
        physics_engine=scenario.PhysicsEngine_dart,
        world: str = None,
        **kwargs,
    ):

        # Compute the number of physics iteration to execute at every environment step
        num_of_steps_per_run = physics_rate / agent_rate

        if num_of_steps_per_run != int(num_of_steps_per_run):
            logger.warn(
                "Rounding the number of iterations to {} from the nominal {}".format(
                    int(num_of_steps_per_run), num_of_steps_per_run
                )
            )

        self.num_of_steps_per_run = int(num_of_steps_per_run)

        # Wrap the task with the runtime
        super().__init__(task_cls=task_cls, physics_rate=physics_rate, agent_rate=agent_rate, real_time_factor=real_time_factor, **kwargs)


    # =================
    # gym.Env interface
    # =================

    def step(self, action: Action) -> State:

        if not self.action_space.contains(action):
            logger.warn("The action does not belong to the action space")

        for sub_step in range(self.num_of_steps_per_run):
            # Set the action
            self.task.set_action(action, store_action=(
                                 sub_step == self.num_of_steps_per_run - 1))

            # Step the simulator
            ok_gazebo = self.gazebo.run()
            assert ok_gazebo, "Failed to step gazebo"

        # Get the observation
        observation = self.task.get_observation()
        assert isinstance(observation, np.ndarray)

        if not self.observation_space.contains(observation):
            logger.warn(
                "The observation does not belong to the observation space")

        # Get the reward
        reward = self.task.get_reward()
        assert isinstance(reward, float), "Failed to get the reward"

        # Check termination
        done = self.task.is_done()

        # Get info
        info = self.task.get_info()

        return State((Observation(observation), Reward(reward), Done(done), Info(info)))

    # ==============================
    # Properties and Private Methods
    # ==============================

    @property
    def gazebo(self) -> scenario.GazeboSimulator:

        if self._gazebo is not None:
            assert self._gazebo.initialized()
            return self._gazebo

        # Create the simulator
        gazebo = scenario.GazeboSimulator(
            1.0 / self._physics_rate,
            self._real_time_factor,
            steps_per_run=1)

        # Store the simulator
        self._gazebo = gazebo

        # Insert the world (it will initialize the simulator)
        _ = self.world
        assert self._gazebo.initialized()

        return self._gazebo
