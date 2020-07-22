import sys

import numpy as np
import pyopencl as cl
import time

xyz = np.dtype([("x", np.int8), ("y", np.int8), ("z", np.int8)])
xyzw = np.dtype([("x", np.int8), ("y", np.int8), ("z", np.int8), ("w", np.int8)])

def correlate(xyz_array, xyzw_array):
    ctx = cl.create_some_context()
    xyz_dtype, xyz_c_decl = cl.tools.match_dtype_to_c_struct(
        ctx.devices[0], 'xyz', xyz)
    xyzw_dtype, xyzw_c_decl = cl.tools.match_dtype_to_c_struct(
        ctx.devices[0], 'xyzw', xyzw)
    xyzw_distance_dtype, xyzw_distance_c_decl = cl.tools.match_dtype_to_c_struct(
        ctx.devices[0], 'xyzw_distance', xyzw)
    cl.tools.get_or_register_dtype('xyz', xyz_dtype)
    cl.tools.get_or_register_dtype('xyzw', xyzw_dtype)
    cl.tools.get_or_register_dtype('xyzw_distance', xyzw_distance_dtype)
    CL_CODE = xyz_c_decl + xyzw_c_decl + xyzw_distance_c_decl + '''
        kernel void correlate(global xyz* xyz_array, global xyzw* xyzw_array, uint xyzw_size, 
                              global xyzw* output_buffer) {
            uint xyz_position = get_global_id(0);
            int gtid = get_global_id(0);
            int ltid = get_local_id(0);
            __local uint my_local_array[8];
            xyz target = xyz_array[xyz_position];
            int w = 0;
            uint min_distance = 4294967295;
            for (int i = 0; i < xyzw_size; i++) {
                xyzw current = xyzw_array[i];
                uint this_distance = sqrt((
                    pow((double)(target.x - current.x), 2) +
                    pow((double)(target.y - current.y), 2) +
                    pow((double)(target.z - current.z), 2))
                );
                if (this_distance < min_distance) {
                    min_distance = this_distance;
                    w = current.w;
                }
            };
            xyzw result = {target.x, target.y, target.z, w};
            output_buffer[xyz_position] = result;
        }
    '''

    prg = cl.Program(ctx, CL_CODE).build()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    local_workgroup_size = min(256, ctx.devices[0].max_work_group_size)
    orig_buffer_length = len(xyz_array)
    padded_buffer_length = np.math.ceil(orig_buffer_length / local_workgroup_size) * local_workgroup_size
    padded_input = np.resize(xyz_array, (padded_buffer_length,))
    xyzw_array_len = np.uint(len(xyzw_array))
    result_np = np.empty_like(xyz_array, dtype=xyzw_dtype)

    xyz_array_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=padded_input)
    xyzw_array_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xyzw_array)
    xyz_result_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=result_np)


    print('beginning correlation')
    prg.correlate(queue, (padded_buffer_length,), (local_workgroup_size,), xyz_array_buf, xyzw_array_buf, xyzw_array_len, xyz_result_buf)
    cl.enqueue_copy(queue, result_np, xyz_result_buf)
    queue.finish()

    return result_np[:orig_buffer_length]


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: correlate_opencl.py /path/to/xyz-file.csv /path/to/xyzw-file.csv /path/to/output.csv")
        exit(-1)

    xyz_arr = np.genfromtxt(sys.argv[1], skip_header=1, dtype=xyz, delimiter=',')
    xyzw_arr = np.genfromtxt(sys.argv[2], skip_header=1, dtype=xyzw, delimiter=',')
    output = sys.argv[3]
    start_time = time.time()
    result = correlate(xyz_arr, xyzw_arr)
    with open(output, mode='w') as f:
        np.savetxt(f, result, header='x,y,z,w', fmt='%d', delimiter=',', comments='')

    print("--- Correlated %d x %d points in %.4f seconds ---" % (len(xyz_arr), len(xyzw_arr), time.time() - start_time))
