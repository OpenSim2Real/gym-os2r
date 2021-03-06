import gym
import time
import numpy as np
from gym_ignition.utils import logger
from gym_os2r import randomizers
from gym_os2r.common import make_mp_envs
import multiprocessing
import sys
import os

# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)


def main_loop(envs):
    current_cumulative_rewards = np.zeros(NUM_ENVS)
    epoch = 0
    beg_time = time.time()
    while epoch < 1000:
        # Execute random actions for each env
        actions = np.stack([envs.action_space.sample()
                            for _ in range(NUM_ENVS)])
        observation_arr, reward_arr, done_arr, _ = envs.step(actions)
        print(envs.get_state_info(observation_arr, actions))
        if any(done_arr):
            print(
                f"{done_arr} ... their reward: {current_cumulative_rewards[done_arr]}")
            current_cumulative_rewards[done_arr] = 0
            epoch += sum(done_arr)
        current_cumulative_rewards += reward_arr
    print('time for 10000 episodes: ' + str(time.time()-beg_time))
    envs.close()
    time.sleep(5)


if __name__ == '__main__':
    try:
        NUM_ENVS = multiprocessing.cpu_count()
        NUMBER_TIME_STEPS = 10000
        seed = 42
        # Available tasks
        env_id = "Monopod-balance-v1"

        # Create a partial function passing the environment id
        kwargs = {'task_mode': 'fixed_hip'}
        envs = make_mp_envs(env_id, NUM_ENVS, seed,
                            randomizers.monopod.MonopodEnvRandomizer, **kwargs)
        envs.reset()
        main_loop(envs)

    except:
        try:
            try:
                envs.close()
            except:
                pass
            sys.exit(0)
        except SystemExit:
            os._exit(0)
