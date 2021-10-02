from gym_ignition.utils.typing import Reward, Observation


"""
Balancing tasks. Start from standing and stay standing.
"""


def balancing_v1(obs: Observation) -> Reward:
    # Get vertical boom angle and velocity.
    bp = obs[2]
    return bp


"""
Standing tasks. Start from ground and stand up.
"""


def standing_v1(obs: Observation) -> Reward:
    # Get vertical boom angle and velocity.
    bp = obs[2]
    return bp
