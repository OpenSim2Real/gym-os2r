# Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import abc
from typing import Union
from scenario import gazebo as scenario
from gym_ignition import utils
from gym_ignition.utils import misc
from gym_ignition import randomizers
from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from gym_ignition.randomizers.model.sdf import Method, Distribution, UniformParams
from gym_ignition.utils.typing import Observation

from gym_bb import tasks
from gym_bb.models import monopod
import random
from gym_bb.rewards.rewards import RewardBase

# Tasks that are supported by this randomizer. Used for type hinting.
SupportedTasks = Union[tasks.monopod_builder.MonopodBuilder]


class MonopodRandomizersMixin(randomizers.abc.TaskRandomizer,
                              randomizers.abc.PhysicsRandomizer,
                              randomizers.abc.ModelDescriptionRandomizer,
                              abc.ABC):
    """
    Mixin that collects the implementation of task, model and physics
    randomizations for monopod environments.
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

        # TODO: Make the reset position adjust to the Reward method.
        # Get the model
        model = task.world.get_model(task.model_name)

        pos_reset = vel_reset = [0]*len(task.joint_names)
        pos_reset[task.joint_names.index(
            'boom_pitch_joint')] = 0.3

        ok_pos = model.to_gazebo().reset_joint_positions(
            pos_reset, task.joint_names)
        ok_vel = model.to_gazebo().reset_joint_velocities(
            vel_reset, task.joint_names)

        if not (ok_pos and ok_vel):
            raise RuntimeError("Failed to reset the monopod state")

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

        # Check env supports at least one model
        if not len(task.supported_models):
            raise RuntimeError('No monopod models support by environement...')
        # Get the model file
        simp_model_name = random.choice(task.supported_models)
        urdf_model_file = monopod.get_model_file_from_name(simp_model_name)

        # Convert the URDF to SDF
        sdf_model_string = scenario.urdffile_to_sdfstring(urdf_model_file)

        # Write the SDF string to a temp file
        sdf_model = utils.misc.string_to_file(sdf_model_string)

        # Create and initialize the randomizer
        sdf_randomizer = randomizers.model.sdf.SDFRandomizer(
            sdf_model=sdf_model)

        # Use the RNG of the task
        sdf_randomizer.rng = task.np_random

        # Add the missing friction/ode/mu element. We assume that friction exists.
        # frictions = randomizer.find_xpath("*/link/collision/surface/friction")
        #
        # for friction in frictions:
        #     # Create parent 'ode' first
        #     if friction.find("ode") is None:
        #         etree.SubElement(friction, "ode")
        #
        #     # Create child 'mu' after
        #     ode = friction.find("ode")
        #     if ode.find("mu") is None:
        #         etree.SubElement(ode, "mu")
        #
        #     # Assign a dummy value to mu
        #     mu = ode.find("mu")
        #     mu.text = str(0)

        randomization_config = {
            "*/link/inertial/mass": {
                # mass + U(-0.5, 0.5)
                'method': Method.Coefficient,
                'distribution': Distribution.Uniform,
                'params': UniformParams(low=0.8, high=1.2),
                'ignore_zeros': True,
                'force_positive': True,
            },
            "*/joint/axis/dynamics/friction": {
                # friction in [0, 5]
                'method': Method.Absolute,
                'distribution': Distribution.Uniform,
                'params': UniformParams(low=0.1, high=0.3),
                'ignore_zeros': False,  # We initialized the value as 0
                'force_positive': True,
            },
            "*/joint/axis/dynamics/damping": {
                # damping (= 3.0) * [0.8, 1.2]
                'method': Method.Coefficient,
                'distribution': Distribution.Uniform,
                'params': UniformParams(low=0.8, high=1.2),
                'ignore_zeros': True,
                'force_positive': True,
            }
        }

        for xpath, config in randomization_config.items():
            sdf_randomizer.new_randomization() \
                .at_xpath(xpath) \
                .method(config["method"]) \
                .sampled_from(config["distribution"], config['params']) \
                .force_positive(config["distribution"]) \
                .ignore_zeros(config["ignore_zeros"]) \
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
                raise RuntimeError(
                    "Failed to remove the monopod from the world")

    @staticmethod
    def _populate_world(task: SupportedTasks, monopod_model: str = None) -> None:

        # Insert a new monopod.
        # It will create a unique name if there are clashing.
        simp_model_name = random.choice(task.supported_models)
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
                 reward_class: RewardBase,
                 num_physics_rollouts: int = 0,
                 **kwargs
                 ):
        # Initialize the mixin
        MonopodRandomizersMixin.__init__(
            self, randomize_physics_after_rollouts=num_physics_rollouts)

        # Initialize the environment randomizer
        gazebo_env_randomizer.GazeboEnvRandomizer.__init__(self, env=env, physics_randomizer=self,
                                                           reward_class=reward_class, **kwargs)

    def get_state_info(self, state: Observation):
        return self.env.unwrapped.task.get_state_info(state)
