import numpy as np


def diffphase(phase_unwrap):
    # Add a 0 in front of the phase_unwrap array to facilitate the calculation of the difference between adjacent phases.
    phase_unwrap = np.concatenate(([0], phase_unwrap))
    len_ = len(phase_unwrap)
    temp = np.zeros(len_ - 1)
    for i in range(1, len_):
        temp[i - 1] = phase_unwrap[i] - phase_unwrap[i - 1]
    return temp

