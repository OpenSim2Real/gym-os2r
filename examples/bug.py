from gym_ignition.utils import logger
import gym
import functools
from gym_bb import randomizers

from gym_bb.common.make_envs import make_env_from_id

env_id = "Monopod-balance-v1"

logger.set_level(gym.logger.DEBUG)

make_env = functools.partial(make_env_from_id, env_id=env_id)

env = randomizers.monopod.MonopodEnvRandomizer(env=make_env)
env.seed(42)

# Try to reset multiple times
action = env.action_space.sample()
print(env.reset())
print(env.step(action))
print(env.reset())
print(env.reset())
print(env.step(action))
print(env.reset())
print(env.step(action))
print(env.step(action))
