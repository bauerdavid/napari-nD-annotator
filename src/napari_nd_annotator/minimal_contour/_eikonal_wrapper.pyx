# cython: boundscheck = False
cimport cython
import numpy as np
cimport numpy as np
cimport openmp
from libcpp cimport bool
from libcpp.vector cimport vector
from cython.operator cimport preincrement as inc
from cython.parallel cimport prange
np.import_array()

cdef extern from "Eikonal.cpp":
    pass

cdef extern from "Eikonal.h":
    cdef cppclass CVec2:
        CVec2() nogil
        CVec2(double, double) nogil
        double x, y
    cdef cppclass SWorkImg[T]:
        SWorkImg()
        void Set(int, int)
        T* operator[](int)
    cdef cppclass SControl:
        SControl() nogil
        int GetProgress() nogil
        int SetParam(int) nogil
        int SetParam(int, int) nogil
        bool DefineInputSet(const vector[CVec2]&, const vector[int]&) nogil
        void SetDataTerm(SWorkImg[double]*) nogil
        void SetDataTerm(SWorkImg[double]*, SWorkImg[double]*) nogil
        void GetDataTerm(SWorkImg[double]**) nogil
        void GetDataTerm(SWorkImg[double]**, SWorkImg[double]**) nogil
        void InitEnvironment(SWorkImg[double]&, SWorkImg[double]&, SWorkImg[double]&) nogil
        int SetNextStartStop() nogil
        void DistanceCalculator() nogil
        int GetReady() nogil
        vector[CVec2]& ResolvePath() nogil
        vector[CVec2]& GetMinPath() nogil

cdef class MinimalContourCalculator:
    # points are as [x, y]
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef run(self, np.ndarray[np.float_t, ndim=3] image, np.ndarray[np.double_t, ndim=2] points, int param=5, reverse_coordinates=False, close_path=True, return_segment_list=False):
        if image.shape[2] != 3:
            print("image should have 3 channels")
            return

        if points.shape[1] != 2:
            print("Points should be 2D")
            return

        cdef int point_count = points.shape[0]
        cdef bool c_close_path = <bool>close_path
        if point_count < 2:
            print("At least two points should be provided")
            return
        cdef bool cancel = False

        cdef vector[CVec2] epoints
        epoints.reserve(point_count)
        cdef vector[int] emethods
        emethods.reserve(point_count)
        cdef int idx = 0
        cdef int i, j
        cdef int X, Y
        if reverse_coordinates:
            X = 1
            Y = 0
        else:
            X = 0
            Y = 1
        for i in range(point_count):
            # TODO Check if point is out of bounds
            epoints.push_back(CVec2(points[i, X], points[i, Y]))
            emethods.push_back(0)
            inc(idx)
        cdef int w = image.shape[1]
        cdef int h = image.shape[0]
        cdef SWorkImg[double] ered, egreen, eblue
        ered.Set(w, h)
        egreen.Set(w, h)
        eblue.Set(w, h)
        cdef double rgb_scale = 1./255.
        cdef int x, y
        for y in range(h):
            r_ptr = ered[y]
            g_ptr = egreen[y]
            b_ptr = eblue[y]
            for x in range(w):
                r_ptr[x] = image[y, x, 0]
                g_ptr[x] = image[y, x, 1]
                b_ptr[x] = image[y, x, 2]
        cdef vector[int] progresses = vector[int](point_count)
        for i in range(point_count):
            progresses[i] = 0
        cdef vector[SControl*] eikonals
        eikonals.reserve(point_count)
        cdef SControl* control
        for i in range(point_count):
            control = new SControl()
            eikonals.push_back(control)

        cdef SWorkImg[double]* split = NULL
        cdef SWorkImg[double]* rand0 = NULL
        cdef SWorkImg[double]* rand1 = NULL

        cdef vector[CVec2] point_pair
        cdef vector[int] method_pair
        cdef CVec2 point1
        cdef CVec2 point2
        cdef int method1
        cdef int method2
        cdef int progress
        cdef SControl* eikonal
        # cdef vector[CVec2] path
        cdef vector[vector[CVec2]] polys
        polys.resize(point_count)
        cdef int n_points = 0
        cdef int num_threads = min(point_count, openmp.omp_get_max_threads())
        for i in prange(point_count, nogil=True, num_threads=num_threads):
        # for i in range(point_count):
            if i == point_count -1:
                if not c_close_path:
                    continue
                point1 = epoints[point_count - 1]
                point2 = epoints[0]

                method1 = emethods[point_count - 1]
                method2 = emethods[0]
            else:
                point1 = epoints[i]
                point2 = epoints[i+1]

                method1 = emethods[i]
                method2 = emethods[i+1]
            point_pair = vector[CVec2](2)
            point_pair[0] = point1
            point_pair[1] = point2
            method_pair = vector[int](2)
            method_pair[0] = method1
            method_pair[1] = method2
            eikonal = eikonals[i]
            eikonal.SetParam(param)
            eikonal.SetParam(0, 0)
            eikonal.DefineInputSet(point_pair, method_pair)
            eikonal.SetDataTerm(split)
            eikonal.SetDataTerm(rand0, rand1)
            eikonal.InitEnvironment(ered, egreen, eblue)

            eikonal.GetDataTerm(&split)
            eikonal.GetDataTerm(&rand0, &rand1)

            eikonal.SetNextStartStop()
            # try:
            progress = progresses[i]

            while True:
                if cancel:
                    break
                eikonal.DistanceCalculator()
                progress = eikonal.GetProgress()
                if eikonal.GetReady() <= 0:
                    break

            eikonal.ResolvePath()
            polys[i] = eikonal.GetMinPath()
            del eikonal
        cdef np.ndarray[np.double_t, ndim=2] segment
        if return_segment_list:
            out_list = []
            for i in range(polys.size()):
                poly = polys[i]
                segment = np.empty((poly.size(), 2), np.float64)
                for j in prange(<int>poly.size(), nogil=True):
                    segment[j, X] = poly[j].x
                    segment[j, Y] = poly[j].y
                out_list.append(segment)
            return out_list
        cdef vector[int] offsets
        offsets.reserve(polys.size())
        offsets[0] = 0
        for i in range(polys.size()):
            n_points += polys[i].size()
            if i < 1:
                continue
            offsets[i] = offsets[i-1]+polys[i-1].size()
        cdef np.ndarray[np.double_t, ndim=2] out = np.zeros([n_points, 2], dtype=np.float64)
        cdef double[:, :] out_view = out
        cdef CVec2 p
        cdef int offset
        cdef int poly_size
        for i in prange(<int>polys.size(), nogil=True, num_threads=num_threads):
            poly = polys[i]
            poly_size = poly.size()
            offset = offsets[i]
            for j in range(poly_size):
                p = poly[poly_size - 1 - j]
                out_view[offset+j, X] = p.x
                out_view[offset+j, Y] = p.y
        return out
