# cython: boundscheck = False
import time

cimport cython
import numpy as np
cimport numpy as np
cimport openmp
from libc.string cimport memcpy
from libcpp cimport bool
from libcpp.vector cimport vector
from cython.operator cimport preincrement as inc
from cython.parallel cimport prange
np.import_array()

GRADIENT_BASED = 0
INTENSITY_BASED = 2
CUSTOM_FEATURE = 3

cdef np.ndarray EMPTY = np.empty(0)

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
        int GetWidth()
        int GetHeight()
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
        void InitEnvironmentAllMethods(SWorkImg[double]&, SWorkImg[double]&, SWorkImg[double]&) nogil
        void InitEnvironmentAllMethods(SWorkImg[double]&, SWorkImg[double]&, SWorkImg[double]&, SWorkImg[double]&, SWorkImg[double]&) nogil
        void InitEnvironmentRanders(SWorkImg[double]&, SWorkImg[double]&, SWorkImg[double]&) nogil
        void InitEnvironmentInhomog(SWorkImg[double]&, SWorkImg[double]&, SWorkImg[double]&) nogil
        void InitEnvironmentRandersGrad(SWorkImg[double]&, SWorkImg[double]&) nogil
        int SetNextStartStop() nogil
        void SetBoundaries(int, int, int, int) nogil
        void SetParAll() nogil
        void DistanceCalculator() nogil
        int GetReady() nogil
        vector[CVec2]& ResolvePath() nogil
        vector[CVec2]& GetMinPath() nogil
        void CleanAll() nogil
        void CalcImageQuantAllMethods() nogil
        # void SetUseLocalMaximum(bool) nogil

cdef class MinimalContourCalculator:
    cdef vector[SControl*] eikonals
    cdef int start_x, start_y, end_x, end_y
    cdef int param
    cdef int method
    cdef vector[int] method_pair
    cdef vector[int] progresses
    cdef SWorkImg[double] ered, egreen, eblue, egradx, egrady
    method_initialized = [False,]*4
    def __cinit__(self, np.ndarray[np.float_t, ndim=3] image, int n_points):
        self.set_image(image, np.empty((0, 0)), np.empty((0, 0)))
        self.eikonals.reserve(n_points)
        self.progresses.resize(n_points)
        self.param = 5
        self.method = 0
        self.method_pair.push_back(self.method)
        self.method_pair.push_back(self.method)
        cdef SControl* control
        cdef int i
        for i in range(n_points):
            control = new SControl()
            control.SetParam(self.param)
            control.SetParam(0, 0)
            self.eikonals.push_back(control)

    cpdef set_use_local_maximum(self, bool use_local_maximum):
        cdef int i=0
        #for i in range(self.eikonals.size()):
            #self.eikonals[i].SetUseLocalMaximum(use_local_maximum)

    cpdef set_param(self, int param):
        cdef int i
        for i in range(self.eikonals.size()):
            self.eikonals[i].SetParam(param)
            self.eikonals[i].CleanAll()
            self.eikonals[i].CalcImageQuantAllMethods()

    cpdef set_method(self, int method=-1):
        if method == -1:
            method = self.method
        if method not in [GRADIENT_BASED, INTENSITY_BASED]:
            print("method should be one of GRADIENT_BASED(=0) or INTENSITY_BASED(=2)")
            return
        self.method = method
        self.method_pair[0] = self.method_pair[1] = method
        cdef int i
        if not self.method_initialized[method]:
            for i in range(self.eikonals.size()):
                if method == GRADIENT_BASED:
                    self.eikonals[i].InitEnvironmentRandersGrad(self.egradx, self.egrady)
                else:
                    self.eikonals[i].InitEnvironmentInhomog(self.ered, self.egreen, self.eblue)
            self.method_initialized[method] = True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef set_boundaries(self, start_x, start_y, end_x, end_y):
        cdef int i
        for i in range(self.eikonals.size()):
            self.eikonals[i].SetBoundaries(start_x, start_y, end_x, end_y)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef set_image(self, np.ndarray[np.float_t, ndim=3] image, np.ndarray[np.float_t, ndim=2] gradx, np.ndarray[np.float_t, ndim=2] grady):
        if image is None:
            return
        if image.shape[2] != 3:
            print("image should have 3 channels")
            return
        cdef int w = image.shape[1]
        cdef int h = image.shape[0]
        cdef np.ndarray[np.float_t, ndim=1] img_max = image.max(axis=(1, 2))
        if self.ered.GetWidth() != w or self.ered.GetHeight() != h:
            self.ered.Set(w, h)
            self.egreen.Set(w, h)
            self.eblue.Set(w, h)
        cdef double rgb_scale = 1. / 255.
        cdef int x, y
        cdef double* r_ptr
        cdef double* g_ptr
        cdef double* b_ptr
        for y in prange(h, nogil=True):
            r_ptr = self.ered[y]
            g_ptr = self.egreen[y]
            b_ptr = self.eblue[y]
            for x in range(w):
                if y == 0 or y == h - 1 or x == 0 or x == w - 1:
                    r_ptr[x] = img_max[0]
                    g_ptr[x] = img_max[1]
                    b_ptr[x] = img_max[2]
                else:
                    r_ptr[x] = image[y, x, 0]
                    g_ptr[x] = image[y, x, 1]
                    b_ptr[x] = image[y, x, 2]
        cdef int i
        for i in range(len(self.method_initialized)):
            self.method_initialized[i] = False
        cdef float max_x
        cdef float max_y
        if gradx.size and grady.size:
            if self.egradx.GetWidth() != w or self.egrady.GetHeight() != h:
                self.egradx.Set(w, h)
                self.egrady.Set(w, h)
            max_x = gradx.max()
            max_y = grady.max()
            for y in prange(h, nogil=True):
                r_ptr = self.egradx[y]
                g_ptr = self.egrady[y]
                for x in range(w):
                    if y == 0 or y == h-1 or x == 0 or x == w-1:
                        r_ptr[x] = max_x/2
                        g_ptr[x] = max_y/2
                    else:
                        r_ptr[x] = gradx[y, x]
                        g_ptr[x] = grady[y, x]
            for i in range(self.eikonals.size()):
                self.eikonals[i].CleanAll()
                self.eikonals[i].SetParAll()
            self.set_method()
        else:
            for i in range(self.eikonals.size()):
                self.eikonals[i].CleanAll()
                self.eikonals[i].SetParAll()
            self.set_method()

    # points are as [x, y]
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef run(
            self,
            np.ndarray[np.double_t, ndim=2] points,
            reverse_coordinates=False,
            close_path=True,
            return_segment_list=False
    ):
        if points.shape[1] != 2:
            print("Points should be 2D")
            return

        if points.shape[0] != self.eikonals.size():
            print("wrong number of points (%d to %d)" % (points.shape[0], self.eikonals.size()))
            return

        cdef int point_count = self.eikonals.size()
        cdef bool c_close_path = <bool>close_path
        if point_count < 2:
            print("At least two points should be provided")
            return
        cdef bool cancel = False

        cdef vector[CVec2] epoints
        epoints.reserve(point_count)
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
            inc(idx)
        for i in range(point_count):
            self.progresses[i] = 0
        cdef vector[CVec2] point_pair
        cdef CVec2 point1
        cdef CVec2 point2
        cdef int progress
        cdef SControl* eikonal
        cdef vector[vector[CVec2]] polys
        polys.resize(point_count)
        cdef int n_points = 0
        cdef int num_threads = min(point_count, openmp.omp_get_max_threads())
        for i in prange(point_count, nogil=True, num_threads=num_threads):
            if i == point_count -1:
                if not c_close_path:
                    continue
                point1 = epoints[point_count - 1]
                point2 = epoints[0]
            else:
                point1 = epoints[i]
                point2 = epoints[i+1]
            point_pair = vector[CVec2](2)
            point_pair[0] = point1
            point_pair[1] = point2

            eikonal = self.eikonals[i]
            eikonal.DefineInputSet(point_pair, self.method_pair)


            eikonal.SetNextStartStop()
            progress = self.progresses[i]
            while True:
                if cancel:
                    break
                eikonal.DistanceCalculator()
                progress = eikonal.GetProgress()
                if eikonal.GetReady() <= 0:
                    break

            eikonal.ResolvePath()
            polys[i] = eikonal.GetMinPath()
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
