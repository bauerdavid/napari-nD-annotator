cimport numpy as np
cpdef np.ndarray[np.double_t, ndim=2] dt(np.ndarray[np.double_t, ndim=2] points, int order)
cpdef magnitude(np.ndarray[np.double_t, ndim=2] points)
cpdef np.ndarray[np.double_t, ndim=1] innerProduct(np.ndarray[np.double_t, ndim=2] curve1, np.ndarray[np.double_t, ndim=2] curve2)