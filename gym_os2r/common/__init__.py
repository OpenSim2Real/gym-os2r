import gym
import functools
from typing import Union

from .vec_env.subproc_vec_env import SubprocVecEnv
from gym_os2r import randomizers

SupportedRandomizers = Union[randomizers.monopod_no_rand.MonopodEnvNoRandomizer,
                             randomizers.monopod.MonopodEnvRandomizer]


def make_env_from_id(env_id: str, **kwargs) -> gym.Env:
    """Utility function for making a single gym env from id.

    Args:
        env_id (str): Environment ID

    Returns:
        env: environment made from env_id

    """
    import gym
    import gym_os2r
    return gym.make(env_id, **kwargs)


def make_mp_envs(env_id,
                 nenvs,
                 seed,
                 randomizer: SupportedRandomizers,
                 start_idx=0,
                 **kwargs):
    """Utility function for making a vec_env from id for multiprocessing.

    Args:
        env_id (str): Environment ID
        nenvs (int): the number of environment you wish to have in subprocesses
        seed (int): the inital seed for RNG
        randomizer (class, 'SupportedRandomizers'): the env randomizer
        rank (int): index of the subprocess
    Returns:
        (:class:`gym_os2r.common.vec_env.SubprocVecEnv`): multiprocessing vectorized environment made from env_id

    """
    def make_env(rank):
        def fn():
            make_env = functools.partial(
                make_env_from_id, env_id=env_id, **kwargs)
            env = randomizer(env=make_env)
            env.seed(seed + rank)
            return env
        return fn
    return SubprocVecEnv([make_env(i + start_idx) for i in range(nenvs)])
