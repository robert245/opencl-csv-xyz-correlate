import sys
import numpy as np
import pandas as pd
import time
from numba import njit, prange

import fish_linearalgebra


def do_correlate(xyz_array, xyzw_arr_dataframe):
    xyzw_array = xyzw_arr_dataframe[['mid_x', 'mid_y', 'mid_z']].to_numpy()

    major_direction = np.array([40.0, 90.0])
    inverse_basis = np.linalg.inv(fish_linearalgebra.calculate_basis_values(major_direction))
    major_anisotropy_power = 5.0
    intermediate_anisotropy_power = 5.0
    minor_anisotropy_power = 1.0

    # Execute the function in parallel
    result = np.zeros([len(xyz_array), 1], dtype=np.int64)
    xyzw_positions = np.array(xyzw_array[:, 0:3], dtype=np.float32)
    correlate_rows(xyz_array[:,0:3], xyzw_positions, inverse_basis,
                   major_anisotropy_power, intermediate_anisotropy_power, minor_anisotropy_power, result)

    return np.append(xyz_array, xyzw_arr_dataframe['M1_LITHOLOGY'].to_numpy()[result], axis=1)


# Use Numbra to JIT compile the below function for a considerable performance boost.
@njit(parallel=True)
def correlate_rows(xyz_array, xyzw_positions,
                   inverse_basis, major_anisotropy_power, intermediate_anisotropy_power, minor_anisotropy_power,
                   result):
    for i in prange(len(xyz_array)):
        current_row = xyz_array[i]
        min_distance = None
        min_pos = 0
        for j in prange(len(xyzw_positions)):
            target = xyzw_positions[j]
            distance = fish_linearalgebra.weight_length(target, current_row, inverse_basis,
                                                        major_anisotropy_power, intermediate_anisotropy_power,
                                                        minor_anisotropy_power)
            if min_distance is None or distance < min_distance:
                min_distance = distance
                min_pos = j

        result[i] = min_pos

    return result


def main():
    if len(sys.argv) != 4:
        print("Usage: correlate_basic.py /path/to/xyz-file.csv /path/to/xyzw-file.csv /path/to/output.csv")
        exit(-1)

    # Get files and output.
    xyz_arr = np.genfromtxt(sys.argv[1], skip_header=1, dtype=np.float32, delimiter=',')
    xyzw_arr_dataframe = pd.read_csv(sys.argv[2], header=0)
    output = sys.argv[3]

    # Record the time, get the result, and write to file.
    start_time = time.time()
    result = do_correlate(xyz_arr, xyzw_arr_dataframe)
    with open(output, mode='w') as f:
        np.savetxt(f, result, header='x,y,z,w', fmt='%s', delimiter=',', comments='')

    print("--- Correlated %d x %d points in %.4f seconds ---" % (len(xyz_arr), len(xyzw_arr_dataframe), time.time() - start_time))


if __name__ == '__main__':
    main()
