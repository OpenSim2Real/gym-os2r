import time
from typing import Union
import random
from BB_gym_Envs import tasks
from BB_gym_Envs.models import monopod
from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from gym_ignition.utils.typing import Action, Reward, Observation

# Tasks that are supported by this randomizer. Used for type hinting.
SupportedTasks = Union[tasks.monopod_v1_balancing.MonopodV1Balancing, \
tasks.monopod_v2_balancing.MonopodV2Balancing, \
tasks.monopod_v1_balancing_fixed_hip.MonopodV1BalancingFixedHip, \
tasks.monopod_v1_balancing_fixed_hip_and_boom_yaw.MonopodV1BalancingFixedHipAndBoomYaw]


class MonopodEnvNoRandomizer(gazebo_env_randomizer.GazeboEnvRandomizer):
    """
    Dummy environment randomizer for monopod tasks.

    Check :py:class:`~BB_gym_Envs.randomizers.monopod.MonopodRandomizersMixin`
    for an example that randomizes the task, the physics, and the model.
    """

    def __init__(self, env: MakeEnvCallable):

        super().__init__(env=env)

    def randomize_task(self, task: SupportedTasks, **kwargs) -> None:
        """
        Prepare the scene for monopod tasks. It simply removes the monopod of the
        previous rollout and inserts a new one in the default state. Then, the active
        Task will reset the state of the monopod depending on the implemented
        decision-making logic.
        """

        if "gazebo" not in kwargs:
            raise ValueError("gazebo kwarg not passed to the task randomizer")

        gazebo = kwargs["gazebo"]

        # Remove the model from the simulation
        if task.model_name is not None and task.model_name in task.world.model_names():
            if not task.world.to_gazebo().remove_model(task.model_name):
                raise RuntimeError("Failed to remove the monopod from the world")
        # Execute a paused run to process model removal
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

        # Insert a new monopod model (randomally choosen a compatible one)
        model = monopod.Monopod(world=task.world, monopod_version=random.choice(task.simp_model_names))

        # Store the model name in the task
        task.model_name = model.name()

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    def get_state_info(self, state: Observation):
        return self.env.unwrapped.task.get_state_info(state)
