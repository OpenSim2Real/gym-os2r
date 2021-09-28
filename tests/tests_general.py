import gym
import time
import functools
import numpy as np
from gym_ignition.utils import logger
from gym_bb import randomizers
from gym_bb.common.mp_env import make_mp_envs, make_env_from_id
from gym_bb.monitor.monitor import VecMonitor, VecMonitorPlot

def single_process():
    env_id = "Monopod-Gazebo-v1"
    make_env = functools.partial(make_env_from_id, env_id=env_id)
    env = randomizers.monopod_no_rand.MonopodEnvNoRandomizer(env=make_env)
    env.seed(42)
    observation = env.reset()

    assert len(observation) == 10, "base monopod should have 10 observations"

    assert env.get_state_info(observation)[1] == False, "Should not need reset after getting reset."

    action = env.action_space.sample()
    observation_after_step, reward, done, _ = env.step(action)

    assert env.get_state_info(observation_after_step)[0] == reward, "should have same reward from step and get state info."
    assert all(observation_after_step != observation), "should have different observation after step."

def single_process_fixed_hip():
    env_id = "Monopod-Gazebo-fh-v1"
    make_env = functools.partial(make_env_from_id, env_id=env_id)
    env = randomizers.monopod_no_rand.MonopodEnvNoRandomizer(env=make_env)
    env.seed(42)
    observation = env.reset()

    assert len(observation) == 8, "Fixed hip monopod should have 8 observations"

    assert env.get_state_info(observation)[1] == False, "Should not need reset after getting reset."

    action = env.action_space.sample()
    observation_after_step, reward, done, _ = env.step(action)
    assert env.get_state_info(observation_after_step)[0] == reward, "should have same reward from step and get state info."
    assert all(observation_after_step != observation), "should have different observation after step."

def test_monopod_model():
    env_id = "Monopod-Gazebo-v1"
    make_env = functools.partial(make_env_from_id, env_id=env_id)
    env = randomizers.monopod.MonopodEnvRandomizer(env=make_env)
    env.seed(42)
    observation = env.reset()
    assert len(observation) == 10, "base monopod should have 10 observations"

    env_id = "Monopod-Gazebo-fh-v1"
    make_env = functools.partial(make_env_from_id, env_id=env_id)
    env = randomizers.monopod.MonopodEnvRandomizer(env=make_env)
    observation = env.reset()
    assert len(observation) == 8, "fixed hip monopod should have 8 observations"

    env_id = "Monopod-Gazebo-fh-fby-v1"
    make_env = functools.partial(make_env_from_id, env_id=env_id)
    env = randomizers.monopod.MonopodEnvRandomizer(env=make_env)
    observation = env.reset()
    assert len(observation) == 6, "fixed hip and fixed boom yaw monopod should have 6 observations"

# def multi_process_fixed_hip():
    # env_id = "Monopod-Gazebo-v1"
    # make_env = functools.partial(make_env_from_id, env_id=env_id)
    # env = randomizers.monopod_no_rand.MonopodEnvNoRandomizer(env=make_env)
    # env.seed(42)
    # observation = env.reset()
    #
    # assert len(observation) == 8, "Fixed hip monopod should have 8 observations"
    #
    # assert env.get_state_info(observation)[1] == False, "Should not need reset after getting reset."
    #
    # action = env.action_space.sample()
    # observation_after_step, reward, done, _ = env.step(action)
    #
    # assert env.get_state_info(observation_after_step)[0] == reward, "should have same reward from step and get state info."
    # assert observation_after_step != observation, "should have different observation after step."



if __name__ == "__main__":
    test_monopod_model()
    single_process()
    single_process_fixed_hip()
    print("Everything passed")
