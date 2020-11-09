import math
import time

import numpy as np
from numba import jit

RAD_CONST = math.pi / 180.0


@jit
def normal(dip, dip_direction):
    nx = math.sin(RAD_CONST * dip) * math.sin(RAD_CONST * dip_direction)
    ny = math.sin(RAD_CONST * dip) * math.cos(RAD_CONST * dip_direction)
    nz = math.cos(RAD_CONST * dip)

    return np.array([nx, ny, nz])


def calculate_basis_values(major_direction):
    major_basis_1 = math.sin(RAD_CONST * major_direction[1]) * math.cos(RAD_CONST * major_direction[0])
    major_basis_2 = math.cos(RAD_CONST * major_direction[1]) * math.cos(RAD_CONST * major_direction[0])
    major_basis_3 = -1 * math.sin(RAD_CONST * major_direction[0])

    major_basis = np.array([major_basis_1, major_basis_2, major_basis_3])
    minor_basis = normal(major_direction[0], major_direction[1])
    intermediate_basis = np.cross(major_basis, minor_basis)

    return np.array([major_basis, intermediate_basis, minor_basis])

@jit
def weight_length(lhs, rhs, inverse_basis,
                  major_anisotropy_power, intermediate_anisotropy_power, minor_anisotropy_power):
    """
    Calculate the weighted length between two 1x3 numpy arrays
    TODO Aidan help me describe what's happening here ;)
    """
    distance_diff = lhs - rhs

    major_length = __calculate_weighted_anisotropy_length(distance_diff,
                                                          inverse_basis[:, 0], major_anisotropy_power)
    intermediate_length = __calculate_weighted_anisotropy_length(distance_diff, inverse_basis[:, 1],
                                                                 intermediate_anisotropy_power)
    minor_length = __calculate_weighted_anisotropy_length(distance_diff,
                                                          inverse_basis[:, 2], minor_anisotropy_power)

    return math.sqrt(major_length ** 2 + intermediate_length ** 2 + minor_length ** 2)


@jit
def __calculate_weighted_anisotropy_length(comparison, inverse_basis_values, anisotropy_power):
    """
    TODO Aidan: not sure what is a sensible name for these variables
    """
    return (inverse_basis_values[0] * comparison[0] +
            inverse_basis_values[1] * comparison[1] +
            inverse_basis_values[2] * comparison[2]) / anisotropy_power


def for_array(target):
    # For testing performance - remove me
    start_time = time.time()
    for t in target:
        weight_length(t)

    print("\n--- Done in %.4f seconds ---" % (time.time() - start_time))
    print('done')


if __name__ == '__main__':
    for_array(np.random.rand(5000000, 6))
