import gym
import functools
from gym_bb import randomizers

env_id = "Monopod-v1"


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    import gym
    import gym_bb
    return gym.make(env_id, **kwargs)


make_env = functools.partial(make_env_from_id, env_id=env_id)

env = randomizers.monopod.MonopodEnvRandomizer(
    env=make_env, reward_class_name='BalancingV1')
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
