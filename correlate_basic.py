import sys
import numpy as np
import time
import math
from multiprocessing import cpu_count, Pool
import functools
from numba import jit


def do_correlate(xyz_array, xyzw_array):

    # Create a thread pool (number of CPU threads, -1 spare for everything else)
    thread_pool = Pool(cpu_count() - 1)

    # Execute the function in parallel
    result = thread_pool.map(functools.partial(correlate_row, xyzw_array=xyzw_array), xyz_array)
    return result


@jit
def correlate_row(current_row, xyzw_array):
    min_distance = None
    current_min = None
    for target in xyzw_array:
        # Take the dif_x, dif_y, dif_z out as we will need these later
        # when we expand the below calculation to include other data points.
        dif_x = target[0] - current_row[0]
        dif_y = target[1] - current_row[1]
        dif_z = target[2] - current_row[2]

        # Take the Euclidean distance of the two dot points - this is also the == l2 norm of current - target
        # distance = np.linalg.norm(current_row - target[0:3])
        distance = math.sqrt(
            math.pow(dif_x, 2) +
            math.pow(dif_y, 2) +
            math.pow(dif_z, 2)
        )
        if min_distance is None or distance < min_distance:
            min_distance = distance
            current_min = target

    return np.append(current_row, current_min[3])


def main():
    if len(sys.argv) != 4:
        print("Usage: correlate_basic.py /path/to/xyz-file.csv /path/to/xyzw-file.csv /path/to/output.csv")
        exit(-1)

    # Get files and output.
    xyz_arr = np.genfromtxt(sys.argv[1], skip_header=1, dtype=np.int8, delimiter=',')
    xyzw_arr = np.genfromtxt(sys.argv[2], skip_header=1, dtype=np.int8, delimiter=',')
    output = sys.argv[3]

    # Record the time, get the result, and write to file.
    start_time = time.time()
    result = do_correlate(xyz_arr, xyzw_arr)
    with open(output, mode='w') as f:
        np.savetxt(f, result, header='x,y,z,w', fmt='%d', delimiter=',', comments='')

    print("--- Correlated %d x %d points in %.4f seconds ---" % (len(xyz_arr), len(xyzw_arr), time.time() - start_time))


if __name__ == '__main__':
    main()
