import gym
import time
import functools
import numpy as np
from gym_ignition.utils import logger
from BB_gym_Envs import randomizers
from BB_gym_Envs.common.mp_env import make_mp_envs
from BB_gym_Envs.monitor.monitor import VecMonitor, VecMonitorPlot
import multiprocessing
import os, sys

# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)

def main_loop(envs):
    # Enable the rendering
    # env.render('human')
    current_cumulative_rewards = np.zeros(NUM_ENVS)

    for step in range(NUMBER_TIME_STEPS):

        # Execute random actions for each env
        actions = np.stack([envs.action_space.sample() for _ in range(NUM_ENVS)])
        observation_arr, reward_arr, done_arr, _ = envs.step(actions)

        if any(done_arr):
            print('rollout info: ', envs.do_rollout(observation_arr))
            print(' Real Reward: ', reward_arr)
            # print(f"Step: {step}, {done_arr} ... their reward: {current_cumulative_rewards[done_arr]}")
            current_cumulative_rewards[done_arr] = 0
        current_cumulative_rewards += reward_arr

    envs.close()
    time.sleep(5)

if __name__ == '__main__':
    try:
        # Available tasks
        env_id = "Monopod-Gazebo-v1"
        NUM_ENVS = multiprocessing.cpu_count()
        NUMBER_TIME_STEPS = 10000
        seed = 42

        fenvs = make_mp_envs(env_id, NUM_ENVS, seed, randomizers.monopod.MonopodEnvRandomizer)
        # envs = VecMonitor(envs)
        envs = VecMonitorPlot(fenvs, plot_path=os.path.expanduser('~')+'/Desktop/plot')

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
