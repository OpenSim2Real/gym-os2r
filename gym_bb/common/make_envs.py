import gym
import functools
from typing import Union

from gym_bb.common.vec_env import SubprocVecEnv
from gym_bb import randomizers
from gym_bb.rewards.reward_definition import RewardBase

SupportedRandomizers = Union[randomizers.monopod_no_rand.MonopodEnvNoRandomizer,
                             randomizers.monopod.MonopodEnvRandomizer]


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    """
    Utility function for making a single gym env from id.

    param env_id: (str) the environment ID
    """
    import gym
    import gym_bb
    return gym.make(env_id, **kwargs)


def make_mp_envs(env_id,
                 nenvs,
                 seed,
                 randomizer: SupportedRandomizers,
                 reward_class: RewardBase,
                 start_idx=0,
                 **kwargs):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param nenvs: (int) the number of environment you wish to have in
                        subprocesses
    :param seed: (int) the inital seed for RNG
    :param randomizer: (SupportedRandomizers) the env randomizer
    :param rank: (int) index of the subprocess
    """
    def make_env(rank):
        def fn():
            make_env = functools.partial(make_env_from_id, env_id=env_id)
            env = randomizer(env=make_env, reward_class=reward_class, **kwargs)
            env.seed(seed + rank)
            return env
        return fn
    return SubprocVecEnv([make_env(i + start_idx) for i in range(nenvs)])
