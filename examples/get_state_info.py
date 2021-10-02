import gym
import time
import functools
import numpy as np
from gym_ignition.utils import logger
from gym_bb import randomizers
from gym_bb.common.make_envs import make_mp_envs
import multiprocessing
import os
import sys

# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)


def main_loop(envs, num_envs):
    NUMBER_TIME_STEPS = 10000
    # Enable the rendering
    envs.reset()

    current_cumulative_rewards = np.zeros(num_envs)
    for step in range(NUMBER_TIME_STEPS):

        # Execute random actions for each env
        actions = np.stack([envs.action_space.sample()
                            for _ in range(num_envs)])
        observation_arr, reward_arr, done_arr, _ = envs.step(actions)
        if any(done_arr):
            print('state info: ', envs.get_state_info(observation_arr))
            print(' Real info: ', reward_arr, done_arr)

    envs.close()
    time.sleep(5)


if __name__ == '__main__':
    try:
        # Available tasks
        env_id = "Monopod-v2"
        NUM_ENVS = multiprocessing.cpu_count()
        NUM_ENVS = 1
        seed = 42

        envs = make_mp_envs(env_id, NUM_ENVS, seed,
                            randomizers.monopod.MonopodEnvRandomizer,
                            reward_type='Balancing_v1')
        main_loop(envs, NUM_ENVS)

    except Exception as error:
        print(error)
        try:
            try:
                envs.close()
            except:
                pass
            sys.exit(0)
        except SystemExit:
            os._exit(0)
