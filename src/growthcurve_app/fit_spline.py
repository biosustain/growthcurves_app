from collections import namedtuple

import numpy as np

SmoothingRange = namedtuple("SmoothingRange", ["s_min", "s", "s_max"])


def get_smoothing_range(m: int):
    """
    Compute the smoothing range for B-spline fitting in scipy interpolate functionality.
    """
    s_min, s, s_max = int(m - np.sqrt(2 * m)), m, int(m + np.sqrt(2 * m))
    s = SmoothingRange(s_min, s, s_max)
    return s
