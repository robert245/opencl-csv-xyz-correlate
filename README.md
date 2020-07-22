# opencl-csv-xyz-correlate

An OpenCL (pyopencl) implementation of https://github.com/robert245/csv-xyz-correlate.

I have less experience writing python vs nodejs, but thought it'd be fun to reimplement the solution using OpenCL 
(Graphics card acceleration).

This is just for fun and the same caveat as before applies - this is not commercial quality code.

# Usage
Install pyopencl and numpy into your environment, perhaps using Conda.

Call the file, passing xyz, xyzw and the path to the result as parameters.

e.g
```shell script
python correlate_opencl.py xyz-10240.csv xyzw.csv output.csv
```

You can use the test files from https://github.com/robert245/csv-xyz-correlate.

As mentioned before, this is just for fun.  There are no guarantees that this code is bug free.  
Use it however you want, but at your own risk