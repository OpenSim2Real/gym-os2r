# Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import abc
from typing import Union

import random
import os
from lxml import etree
from operator import add
from functools import reduce

from scenario import gazebo as scenario
from gym_ignition import utils
from gym_ignition.utils import misc
from gym_ignition import randomizers
from gym_ignition.randomizers import gazebo_env_randomizer
from gym_ignition.randomizers.gazebo_env_randomizer import MakeEnvCallable
from gym_ignition.randomizers.model.sdf import Method, Distribution, UniformParams
from gym_ignition.utils.typing import Observation, Action
from gym_ignition.utils import logger

from gym_bb import tasks
from gym_bb.models import monopod
from gym_bb.utils.reset import leg_joint_angles

# Tasks that are supported by this randomizer. Used for type hinting.
SupportedTasks = Union[tasks.monopod.MonopodTask]


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

        reset_position = random.choice(task.reset_positions)
        xpath = 'resets/' + reset_position
        reset_conf = task.cfg.get_config(xpath)
        # Randomization,
        reset_conf['boom_pitch_joint'] *= random.uniform(0.8, 1.2)
        joint_angles = (0, 0)
        if not reset_conf['laying_down']:
            xpath = 'task_modes/' + task.task_mode + '/definition'
            robot_def = task.cfg.get_config(xpath)
            robot_def['boom_pitch_joint'] = reset_conf['boom_pitch_joint']
            joint_angles = leg_joint_angles(robot_def)
        else:
            joint_angles = (1.57,  0)
        random_dir = random.choice([-1, 1])
        joint_angles = [angle * random_dir for angle in joint_angles]
        # Get the model
        model = task.world.get_model(task.model_name)

        pos_reset = [0]*len(task.joint_names)
        vel_reset = [0]*len(task.joint_names)
        pos_reset[task.joint_names.index(
            'boom_pitch_joint')] = reset_conf['boom_pitch_joint']
        pos_reset[task.joint_names.index('upper_leg_joint')] = joint_angles[0]
        pos_reset[task.joint_names.index('lower_leg_joint')] = joint_angles[1]

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

        xpath = 'task_modes/' + task.task_mode + '/model'
        monopod_model = task.cfg.get_config(xpath)
        # Get the model file
        urdf_model_file = monopod.get_model_file_from_name(monopod_model)

        # Convert the URDF to SDF
        sdf_model_string = scenario.urdffile_to_sdfstring(urdf_model_file)

        # Write the SDF string to a temp file
        sdf_model = utils.misc.string_to_file(sdf_model_string)

        # Create and initialize the randomizer
        sdf_randomizer = randomizers.model.sdf.SDFRandomizer(
            sdf_model=sdf_model)

        # Use the RNG of the task
        sdf_randomizer.rng = task.np_random

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
                'method': Method.Absolute,
                'distribution': Distribution.Uniform,
                'params': UniformParams(low=0.01, high=0.1),
                'ignore_zeros': False,  # We initialized the value as 0
                'force_positive': True,
            },
            "*/joint/axis/dynamics/damping": {
                'method': Method.Coefficient,
                'distribution': Distribution.Uniform,
                'params': UniformParams(low=0.8, high=1.2),
                'ignore_zeros': True,
                'force_positive': True,
            },
            "*/link/collision/surface/friction/ode/mu": {
                'method': Method.Absolute,
                'distribution': Distribution.Uniform,
                'params': UniformParams(low=0.8, high=1.2),
                'ignore_zeros': False,
                'force_positive': True,
            }
        }

        def recursive_element_helper(path, randomizer):
            if path in ['', '*']:
                # Return if at base case (last element of xpath)
                return randomizer.find_xpath('*'), []
            path_split = os.path.split(path)
            # Depth first.
            elements, _ = recursive_element_helper(
                path_split[0], randomizer)
            """ Check if current path has elements in it.
             If current path has no elements then make them using one level
             down path. (loop invariant is the level below must exist)"""
            changed = []
            for element in elements:
                if element.find(path_split[1]) is None:
                    etree.SubElement(element, path_split[1])
                    changed.append(element)
                    logger.debug('Added the child ' + str(path_split[1])
                                 + ' to the sdf element ' + str(path_split[0]))
            changed = [element.findall(path_split[1]) for element in changed]
            changed = reduce(add, changed) if changed else changed
            new = reduce(add, [element.findall(path_split[1])
                               for element in elements])
            return new, changed

        def recursive_element_init(path, randomizer):
            elements, changed = recursive_element_helper(path, randomizer)
            for element in changed:
                logger.debug('The leaf element added with the tag: '
                             + str(element.tag) + ' got set to the value 0')
                element.text = str(0)

        for xpath, config in randomization_config.items():
            recursive_element_init(xpath, sdf_randomizer)
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
        xpath = 'task_modes/' + task.task_mode + '/model'
        monopod_model = task.cfg.get_config(xpath)
        model = monopod.Monopod(world=task.world, monopod_version=monopod_model,
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
                 num_physics_rollouts: int = 0,
                 **kwargs
                 ):
        # Initialize the mixin
        MonopodRandomizersMixin.__init__(
            self, randomize_physics_after_rollouts=num_physics_rollouts)

        # Initialize the environment randomizer
        gazebo_env_randomizer.GazeboEnvRandomizer.__init__(self, env=env,
                                                           physics_randomizer=self,
                                                           **kwargs)

    def get_state_info(self, state: Observation, action: Action):
        return self.env.unwrapped.task.get_state_info(state, action)
