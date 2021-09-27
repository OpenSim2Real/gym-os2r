# Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import abc
from typing import Union
from gym_ignition import utils
from gym_ignition.utils import misc
from gym_ignition import randomizers
from scenario import gazebo as scenario
from BB_gym_Envs import tasks
from BB_gym_Envs.models import monopod
from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from gym_ignition.randomizers.model.sdf import Method, Distribution, UniformParams
from gym_ignition.utils.typing import Action, Reward, Observation

# Tasks that are supported by this randomizer. Used for type hinting.
SupportedTasks = Union[tasks.monopod_v1_balancing_fixed_hip.MonopodV1BalancingFixedHip]


simp_model_name = 'monopod_v1_fh'

class MonopodRandomizersMixin(randomizers.abc.TaskRandomizer,
                               randomizers.abc.PhysicsRandomizer,
                               randomizers.abc.ModelDescriptionRandomizer,
                               abc.ABC):
    """
    Mixin that collects the implementation of task, model and physics randomizations for
    monopod environments.
    """

    def __init__(self, randomize_physics_after_rollouts: int = 0):

        # Initialize base classes
        randomizers.abc.TaskRandomizer.__init__(self)
        randomizers.abc.PhysicsRandomizer.__init__(
            self, randomize_after_rollouts_num=randomize_physics_after_rollouts)
        randomizers.abc.ModelDescriptionRandomizer.__init__(self)

        # SDF randomizer
        self._sdf_randomizer = None

    # ===========================
    # PhysicsRandomizer interface
    # ===========================

    def get_engine(self):

        return scenario.PhysicsEngine_dart

    def randomize_physics(self, task: SupportedTasks, **kwargs) -> None:

        gravity_z = task.np_random.normal(loc=-9.8, scale=0.2)

        if not task.world.to_gazebo().set_gravity((0, 0, gravity_z)):
            raise RuntimeError("Failed to set the gravity")

    # ========================
    # TaskRandomizer interface
    # ========================

    def randomize_task(self, task: SupportedTasks, **kwargs) -> None:

        # Remove the model from the world
        self._clean_world(task=task)

        if "gazebo" not in kwargs:
            raise ValueError("gazebo kwarg not passed to the task randomizer")

        gazebo = kwargs["gazebo"]

        # Execute a paused run to process model removal
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

        # Generate a random model description
        random_model = self.randomize_model_description(task=task)

        # Insert a new model in the world
        self._populate_world(task=task, monopod_model=random_model)

        # Execute a paused run to process model insertion
        if not gazebo.run(paused=True):
            raise RuntimeError("Failed to execute a paused Gazebo run")

    # ====================================
    # ModelDescriptionRandomizer interface
    # ====================================

    def randomize_model_description(self, task: SupportedTasks, **kwargs) -> str:

        randomizer = self._get_sdf_randomizer(task=task)
        sdf = misc.string_to_file(randomizer.sample())
        return sdf

    # ===============
    # Private Methods
    # ===============

    def _get_sdf_randomizer(self, task: SupportedTasks) -> \
            randomizers.model.sdf.SDFRandomizer:

        if self._sdf_randomizer is not None:
            return self._sdf_randomizer

        # Get the model file
        urdf_model_file = monopod.get_model_file_from_name(simp_model_name)

        # Convert the URDF to SDF
        sdf_model_string = scenario.urdffile_to_sdfstring(urdf_model_file)

        # Write the SDF string to a temp file
        sdf_model = utils.misc.string_to_file(sdf_model_string)

        # Create and initialize the randomizer
        sdf_randomizer = randomizers.model.sdf.SDFRandomizer(sdf_model=sdf_model)

        # Use the RNG of the task
        sdf_randomizer.rng = task.np_random

        # Randomize the mass of all links
        sdf_randomizer.new_randomization() \
            .at_xpath("*/link/inertial/mass") \
            .method(Method.Additive) \
            .sampled_from(Distribution.Uniform, UniformParams(low=-0.2, high=0.2)) \
            .force_positive() \
            .add()

        # Process the randomization
        sdf_randomizer.process_data()
        assert len(sdf_randomizer.get_active_randomizations()) > 0

        # Store and return the randomizer
        self._sdf_randomizer = sdf_randomizer
        return self._sdf_randomizer

    @staticmethod
    def _clean_world(task: SupportedTasks) -> None:

        # Remove the model from the simulation
        if task.model_name is not None and task.model_name in task.world.model_names():

            if not task.world.to_gazebo().remove_model(task.model_name):
                raise RuntimeError("Failed to remove the monopod from the world")

    @staticmethod
    def _populate_world(task: SupportedTasks, monopod_model: str = None) -> None:

        # Insert a new monopod.
        # It will create a unique name if there are clashing.
        model = monopod.Monopod(world=task.world, monopod_version=simp_model_name,
                                  model_file=monopod_model)

        # Store the model name in the task
        task.model_name = model.name()


class MonopodEnvRandomizer(gazebo_env_randomizer.GazeboEnvRandomizer,
                            MonopodRandomizersMixin):
    """
    Concrete implementation of monopod environments randomization.
    """

    def __init__(self,
                 env: MakeEnvCallable,
                 num_physics_rollouts: int = 0):

        # Initialize the mixin
        MonopodRandomizersMixin.__init__(
            self, randomize_physics_after_rollouts=num_physics_rollouts)

        # Initialize the environment randomizer
        gazebo_env_randomizer.GazeboEnvRandomizer.__init__(
            self, env=env, physics_randomizer=self)

    def get_state_info(self, state: Observation):
        return self.env.unwrapped.task.get_state_info(state)
