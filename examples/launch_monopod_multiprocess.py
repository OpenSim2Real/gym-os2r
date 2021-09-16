import gym
import time
import functools
from gym_ignition.utils import logger
from BB_gym_Envs import randomizers
from BB_gym_Envs.common.mp_env import make_mp_envs

# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)

# Available tasks
env_id = "Monopod-Gazebo-v1"
num_env = 4
seed = 42

# def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
#     import gym
#     import BB_gym_Envs
#     return gym.make(env_id, **kwargs)
#
# make_env = functools.partial(make_env_from_id, env_id=env_id)
# env = randomizers.monopod.MonopodEnvRandomizer(
#     env=make_env)
# env.seed(42)

env = make_mp_envs(env_id, num_env, seed, randomizers.monopod.MonopodEnvRandomizer)

# Enable the rendering
# env.render('human')

for epoch in range(100):

    # Reset the environment
    observation = env.reset()

    # Initialize returned values
    done = False
    totalReward = []

    while not done:
        # Execute a random action
        action = []
        for _ in range(num_env):
            action.append(env.action_space.sample())

        observation_arr, reward_arr, done_arr, _ = env.step(action)

        if totalReward == []:
            totalReward = reward_arr
        else:
            totalReward = [sum(x) for x in zip(totalReward, reward_arr)]

        done = all(done_arr)

    print(f"Reward episode #{epoch}: {totalReward}")


env.close()
time.sleep(5)
