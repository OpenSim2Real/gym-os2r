import numpy as np


def leg_joint_angles(robot_def: dict):
    required = ['boom_pitch_joint', 'upper_leg_length',
                'lower_leg_length', 'central_pivot_height', 'length_boom',
                'hip_offset', 'clipping_adjust']
    if not set(robot_def.keys()).issubset(set(required)):
        raise RuntimeError('One or more of the required params'
                           + str(required)
                           + 'were not provided for finding reset positions. ')
    lb = robot_def['length_boom']
    bp = robot_def['boom_pitch_joint']
    cph = robot_def['central_pivot_height']
    lh = (lb*np.sin(bp)+cph)/np.cos(bp)
    ul = robot_def['upper_leg_length']
    ll = robot_def['lower_leg_length']
    lleg = lh - robot_def['hip_offset'] - robot_def['clipping_adjust']
    # Triangle inequality
    if lleg > ul + ll:
        return (0, 0)
    else:
        upper_leg_angle = np.arccos((ul**2 + lleg**2-ll**2)/(2*ul*lleg))
        lower_leg_angle = np.arcsin(
            ul*np.sin(upper_leg_angle)/ll) + upper_leg_angle
        return (upper_leg_angle, -lower_leg_angle)