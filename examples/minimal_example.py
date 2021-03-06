import gym
import time
import functools

from gym_os2r import randomizers
from gym_os2r.common import make_env_from_id
# from gym_os2r.rewards import BalancingV3

from gym_ignition.utils import logger
# Set verbosity
logger.set_level(gym.logger.ERROR)
# logger.set_level(gym.logger.DEBUG)


# from gym_os2r.rewards import RewardBase
#
# class ExampleV0(RewardBase):
#     """
#     Balancing reward. Start from standing positions and stay standing.
#     """
#
#     def __init__(self, observation_index: dict, normalized: bool):
#         super().__init__(observation_index, normalized)
#         self.supported_task_modes = ['free_hip', 'fixed_hip', 'fixed_hip_torque', 'fixed_hip_simple', 'fixed']
#
#     def calculate_reward(self, obs: Observation, actions: Deque[Action]) -> Reward:
#         return 1
#
# kwargs = {'reward_class': ExampleV0}

# Available tasks
env_id = "Monopod-balance-v3"

# Create a partial function passing the environment id
# kwargs = {'task_mode': 'free_hip'}
# kwargs = {'reset_positions': ['stand', 'ground', 'lay', 'float']}
# kwargs = {'reset_positions': ['stand']}
# kwargs = {'reset_positions': ['float']}

# kwargs = {'task_mode': 'fixed'}
# kwargs = {'reward_class': BalancingV3}

make_env = functools.partial(make_env_from_id, env_id=env_id, **kwargs)

env = randomizers.monopod.MonopodEnvRandomizer(env=make_env)
# env = randomizers.monopod_no_rand.MonopodEnvNoRandomizer(env=make_env)

# Enable the rendering
env.render('human')
# Initialize the seed
env.seed(42)

beg_time = time.time()
for epoch in range(1000):

    # Reset the environment
    observation = env.reset()
    # time.sleep(1)

    # Initialize returned values
    done = False
    totalReward = 0
    count = 0

    time.sleep(1)
    c = 0
    while not done:
        # Execute a random action
        # action = env.action_space.sample() * 0.1  # make the value smaller
        # action = env.action_space.sample()
        action = [-1, -1]
        # action = [0.0, 0.2]
        observation, reward, done, info = env.step(action)
        # done = True
        # time.sleep(0.05)
        c += 1
        done = c==400

    # while not done:
    #     count += 1
    #     # Execute a random action
    #     # action = env.action_space.sample() * 0.1  # make the value smaller
    #     action = env.action_space.sample()
    #     # action = [0.0, 0.2]
    #     observation, reward, done, _ = env.step(action)
    #
    #     if observation[2] < -0.1:
    #         print('Bad obs: ', observation)
    #         done = True
    #     # print('observations: ', observation)
    #     # print('obseration high: ', env.observation_space.high, 'obseration low: ', env.observation_space.low)
    #     # print(env.task.observation_index)
    #     if count == 1000:
    #         done = True

env.close()
time.sleep(5)
