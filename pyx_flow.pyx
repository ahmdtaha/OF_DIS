# distutils: language = c++
# distutils: sources = run_dense.cpp
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
cimport numpy as np
from libc.string cimport memcpy
from opencv_mat cimport *
# Author: Deepak Pathak (c) 2016


## define c extension extracted from from msrc.h for add(int a,int b)
cdef extern from "run_dense.h":
    Mat optical_flow(char* imgfile_ao,char* imgfile_bo);

cdef extern from "opencv2/opencv.hpp":        
    cdef int CV_8UC3
    cdef int CV_8UC1
    cdef int CV_32FC1
    cdef int CV_64FC1


## Define the python wrapper that pass calls to the c++ function add(a,b)
def optical_fn(a,b):
    cdef Mat m
    m = optical_flow(a,b)
    print("reached here");
    cdef Py_buffer buf_info
    # Define the size / len of data
    cdef size_t len = m.rows*m.cols*m.channels()*sizeof(CV_32FC1)
    print(len,m.rows,m.cols,m.channels(),sizeof(CV_32FC1))
    # Fill buffer
    PyBuffer_FillInfo(&buf_info, NULL, m.data, len, 1, PyBUF_FULL_RO)
    Pydata  = PyMemoryView_FromBuffer(&buf_info)
    shape_array = (m.rows, m.cols, m.channels())
    ary = np.ndarray(shape=shape_array, buffer=Pydata, order='c', dtype=np.float32)
    result = np.copy(ary)
    return result





