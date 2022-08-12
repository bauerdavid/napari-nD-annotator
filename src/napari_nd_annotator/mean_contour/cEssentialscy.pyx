import numpy as np
cimport numpy as np
import time
from enum import Enum
from cython.parallel cimport prange
cimport cython
from libc.math cimport sqrt
class ReconstructionMethods(Enum):
    NEWTON = 1
    GRADIENT_DESCENT = 2
    CG = 3
    SKIP = 4
    JOZSI_GRADIENT = 5
    

# derivative approximation
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef np.ndarray[np.double_t, ndim=2] dt(np.ndarray[np.double_t, ndim=2] points, int order):
    cdef int n_points = points.shape[0]
    cdef double next_x_1, next_y_1, next_x_2, next_y_2, next_x_3, next_y_3, next_x_4, next_y_4
    cdef double prev_x_1, prev_y_1, prev_x_2, prev_y_2, prev_x_3, prev_y_3, prev_x_4, prev_y_4
    cdef double prev_factor_x, prev_factor_y, next_factor_x, next_factor_y, curr_factor_x, curr_factor_y
    cdef np.ndarray[np.double_t, ndim=2] out = np.empty_like(points)
    cdef int i
    for i in prange(n_points, nogil=True):
        if i < 4:
            prev_x_4 = points[n_points + i - 4, 0]
            prev_y_4 = points[n_points + i - 4, 1]
            if i < 3:
                prev_x_3 = points[n_points + i - 3, 0]
                prev_y_3 = points[n_points + i - 3, 1]
                if i < 2:
                    prev_x_2 = points[n_points + i - 2, 0]
                    prev_y_2 = points[n_points + i - 2, 1]
                    if i < 1:
                        prev_x_1 = points[n_points + i - 1, 0]
                        prev_y_1 = points[n_points + i - 1, 1]
                    else:
                        prev_x_1 = points[i - 1, 0]
                        prev_y_1 = points[i - 1, 1]
                else:
                    prev_x_2 = points[i - 2, 0]
                    prev_y_2 = points[i - 2, 1]
                    prev_x_1 = points[i - 1, 0]
                    prev_y_1 = points[i - 1, 1]
            else:
                prev_x_3 = points[i - 3, 0]
                prev_y_3 = points[i - 3, 1]
                prev_x_2 = points[i - 2, 0]
                prev_y_2 = points[i - 2, 1]
                prev_x_1 = points[i - 1, 0]
                prev_y_1 = points[i - 1, 1]
        else:
            prev_x_4 = points[i - 4, 0]
            prev_y_4 = points[i - 4, 1]
            prev_x_3 = points[i - 3, 0]
            prev_y_3 = points[i - 3, 1]
            prev_x_2 = points[i - 2, 0]
            prev_y_2 = points[i - 2, 1]
            prev_x_1 = points[i - 1, 0]
            prev_y_1 = points[i - 1, 1]
        if i + 4 >= n_points:
            next_x_4 = points[i+4-n_points, 0]
            next_y_4 = points[i+4-n_points, 1]
            if i + 3 >= n_points:
                next_x_3 = points[i + 3 - n_points, 0]
                next_y_3 = points[i + 3 - n_points, 1]
                if i + 2 >= n_points:
                    next_x_2 = points[i + 2 - n_points, 0]
                    next_y_2 = points[i + 2 - n_points, 1]
                    if i + 1 >= n_points:
                        next_x_1 = points[i + 1 - n_points, 0]
                        next_y_1 = points[i + 1 - n_points, 1]
                    else:
                        next_x_1 = points[i + 1, 0]
                        next_y_1 = points[i + 1, 1]
                else:
                    next_x_2 = points[i + 2, 0]
                    next_y_2 = points[i + 2, 1]
                    next_x_1 = points[i + 1, 0]
                    next_y_1 = points[i + 1, 1]
            else:
                next_x_3 = points[i + 3, 0]
                next_y_3 = points[i + 3, 1]
                next_x_2 = points[i + 2, 0]
                next_y_2 = points[i + 2, 1]
                next_x_1 = points[i + 1, 0]
                next_y_1 = points[i + 1, 1]
        else:
            next_x_4 = points[i+4, 0]
            next_y_4 = points[i+4, 1]
            next_x_3 = points[i + 3, 0]
            next_y_3 = points[i + 3, 1]
            next_x_2 = points[i + 2, 0]
            next_y_2 = points[i + 2, 1]
            next_x_1 = points[i + 1, 0]
            next_y_1 = points[i + 1, 1]
        if order == 1:
            prev_factor_x = (1. / 280.) * prev_x_4 - (4. / 105.) * prev_x_3 + (1. / 5.) * prev_x_2 - (4. / 5.) * prev_x_1
            prev_factor_y = (1. / 280.) * prev_y_4 - (4. / 105.) * prev_y_3 + (1. / 5.) * prev_y_2 - (4. / 5.) * prev_y_1
            next_factor_x = (1. / 280.) * next_x_4 - (4. / 105.) * next_x_3 + (1. / 5.) * next_x_2 - (4. / 5.) * next_x_1
            next_factor_y = (1. / 280.) * next_y_4 - (4. / 105.) * next_y_3 + (1. / 5.) * next_y_2 - (4. / 5.) * next_y_1
            out[i, 0] = prev_factor_x+next_factor_x
            out[i, 1] = prev_factor_y+next_factor_y
        elif order == 2:
            prev_factor_x = (-1./560.)*prev_x_4+(8./315.)*prev_x_3-(1./5.)*prev_x_2+(8./5.)*prev_x_1
            prev_factor_y = (-1./560.)*prev_y_4+(8./315.)*prev_y_3-(1./5.)*prev_y_2+(8./5.)*prev_y_1
            next_factor_x = (-1. / 560.) * next_x_4 + (8. / 315.) * next_x_3 - (1. / 5.) * next_x_2 + (8. / 5.) * next_x_1
            next_factor_y = (-1. / 560.) * next_y_4 + (8. / 315.) * next_y_3 - (1. / 5.) * next_y_2 + (8. / 5.) * next_y_1
            curr_factor_x = (-205. / 72.) * points[i, 0]
            curr_factor_y = (-205. / 72.) * points[i, 1]
            out[i, 0] = prev_factor_x + next_factor_x + curr_factor_x
            out[i, 1] = prev_factor_y + next_factor_y + curr_factor_y
    return out

# magnitude of a single vector or a set of vectors
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef magnitude(np.ndarray[np.double_t, ndim=2] points):
    cdef np.ndarray[np.double_t, ndim=1] out = np.empty(points.shape[0], np.float64)
    cdef int i
    for i in prange(points.shape[0], nogil=True):
        out[i] = sqrt(points[i, 0]*points[i, 0] + points[i, 1]*points[i, 1])
    return out
    # return np.sqrt(points[:,0]*points[:,0]+points[:,1]*points[:,1])
        
# inner product of two curves
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.double_t, ndim=1] innerProduct(np.ndarray[np.double_t, ndim=2] curve1, np.ndarray[np.double_t, ndim=2] curve2):
    # return np.sum(curve1*curve2, axis=1)
    return curve1[:,0]*curve2[:,0]+curve1[:,1]*curve2[:,1]

class Contour:

    def __init__(self, pointArray, nPoi, resMultiplier):
        # number of points in contour
        self.nPoi = pointArray.shape[0]
        self.cPoints = nPoi
        self.resMultiplier = resMultiplier
        self.resolution = self.cPoints*self.resMultiplier
        self.pointArray = pointArray
        self.contourLength = self.getContourLength()
        self.centroid = self.getCentroid()
        self.lookup, self.parameterization = self.getLookupTables()
        self.smoothLookupTable()
        self.diffs = np.empty(self.cPoints)
        self.calcParams()

    def getCentroid(self):
        centroid = np.zeros((1,2))
        for p in self.pointArray:
            centroid = centroid + p
        centroid = centroid/self.nPoi
        return centroid

    def dt(self, order):
        '''nextParam = self.nextParam
        prevParam = self.prevParam
        nextParam2 = np.roll(nextParam,-1,axis=0)
        prevParam2 = np.roll(prevParam,1,axis=0)'''

        if order==1:
            return dt(self.lookup[self.parameterization], 1)
            #return (1/12)*self.lookup[prevParam2]-(2/3)*self.lookup[prevParam]+(2/3)*self.lookup[nextParam]-(1/12)*self.lookup[nextParam]
            #return (self.lookup[nextParam,:]-self.lookup[prevParam,:])/2
        if order==2:
            return dt(self.lookup[self.parameterization], 2)
            #return (-1/12)*self.lookup[prevParam2]+(4/3)*self.lookup[prevParam]-(5/2)*self.lookup[self.parameterization]+(4/3)*self.lookup[nextParam]-(1/12)*self.lookup[nextParam2]
            #return self.lookup[nextParam,:]+self.lookup[prevParam,:]-2*self.lookup[self.parameterization,:]
    
    
    #Christoffel divergence as defined in (9): Gamma_i
    def getChristoffelDivergence(self):
        deriv = self.derivatives
        sderiv = self.sderivatives
        return (deriv[:,0]*sderiv[:,0]+deriv[:,1]*sderiv[:,1])/(deriv[:,0]*deriv[:,0]+deriv[:,1]*deriv[:,1])

    def getSRV(self):
        deriv = self.derivatives
        return np.sqrt(magnitude(deriv)) 

    def getRPSV(self):
        return self.lookup[self.parameterization]*self.srv[:,None]

    def getSRVF(self):
        res = self.derivatives.copy()
        deriv = self.derivatives
        velo = magnitude(deriv)
        srv = np.sqrt(velo)
        res[:,0] /= srv
        res[:,1] /= srv
        return res

    def calcParams(self):
        
        self.nextParam = np.roll(self.parameterization, -1, axis=0)
        self.prevParam = np.roll(self.parameterization, 1, axis=0)

        self.derivatives = self.dt(1)
        self.sderivatives = self.dt(2)

        deriv = self.derivatives
        sderiv = self.sderivatives

        self.christoffel = self.getChristoffelDivergence()
        self.srv = self.getSRV()


    def getIdxDiff(self, dgamma):
        nextParam = self.nextParam
        prevParam = self.prevParam
        criteria1 = dgamma<0
        criteria2 = dgamma>=0

        diffs = self.diffs

        diffs[:] = 0
        
        diffs[criteria1] = self.parameterization[criteria1] - prevParam[criteria1]
        diffCriteria = diffs<0
        diffs[diffCriteria] = self.parameterization[diffCriteria]+self.cPoints*self.resMultiplier-prevParam[diffCriteria]

        diffs[criteria2] = nextParam[criteria2]-self.parameterization[criteria2]
        diffCriteria = diffs<0
        diffs[diffCriteria] = nextParam[diffCriteria]+self.cPoints*self.resMultiplier-self.parameterization[diffCriteria]
        

        return diffs

    # make sure we remain in lookup table index range with the reparameterization
    def getInRangeParameterization(self):
        pointNum = self.lookup.shape[0]
        criteria1 = self.parameterization.copy() >= pointNum
        criteria2 = self.parameterization.copy() < 0
        self.parameterization[criteria2] %= pointNum
        self.parameterization[criteria2] = -1*(self.parameterization[criteria2]-pointNum)
        self.parameterization[criteria1] %= pointNum

    # smoothing to ensure that 
    def smoothParameterization(self):
        resHalf = self.parameterization.shape[0]*self.resMultiplier/2

        tmparam = self.parameterization.copy()
        tmp = self.parameterization.copy()
        tmpnext = np.roll(tmp, -1, axis=0)
        tmpprev = np.roll(tmp, 1, axis=0)
        crit1 = (tmpnext < resHalf) & (tmp > resHalf) & (tmpprev > resHalf)
        tmpnext[crit1] += self.cPoints*self.resMultiplier
        crit2 = (tmp < resHalf) & (tmpprev > resHalf)
        tmp[crit2] += self.cPoints*self.resMultiplier
        tmpnext[crit2] += self.cPoints*self.resMultiplier

        self.parameterization = 2+2*tmp+tmpnext+tmpprev
        self.parameterization >>= 2

        self.getInRangeParameterization()

    def smoothLookupTable(self):
        nextArr = np.roll(self.lookup, -1, axis=0)
        prevArr = np.roll(self.lookup, 1, axis=0)

        temp = 0.25*(2* self.lookup + prevArr + nextArr)
        tempNext = np.roll(temp, -1, axis=0)
        tempPrev = np.roll(temp, 1, axis=0)

        self.lookup = 0.25*(2* temp + tempPrev + tempNext)

    def getContourLength(self):
        nextArr = np.roll(self.pointArray, -1, axis=0)

        cLength = np.sum(np.sqrt((nextArr[:,0]-self.pointArray[:,0])*(nextArr[:,0]-self.pointArray[:,0])+(nextArr[:,1]-self.pointArray[:,1])*(nextArr[:,1]-self.pointArray[:,1])))
        return cLength
    

    def isClockwise(self):
        nextLook = np.roll(self.lookup, -1, axis=0)
        edges = self.lookup[self.parameterization,0]*nextLook[self.parameterization,1]-nextLook[self.parameterization,0]*self.lookup[self.parameterization,1]
        if np.sum(edges)>0:
            return True
        else:
            return False

    def setStartingPointToLowestY(self):
        lowestY = np.argmin(self.lookup[:,1])
        self.lookup = np.roll(self.lookup, -1*lowestY, axis=0)

    def getLookupTables(self):
        start = time.time()
        # length of one step if we want to achieve self.resolution:
        unitLength = self.contourLength/self.resolution
        # lookup table
        lut = np.empty((2*self.resolution,2))
        # shifted point arrays
        nextArray = np.roll(self.pointArray, -1, axis=0)

        j = 0 # index of LookUpTable
        remainder = 0 # length of overflow for sections between 2 points
        sum_while = 0
        for i in range(self.nPoi):
            startPoint = self.pointArray[i]
            nextPoint = nextArray[i]
            direction = nextPoint-startPoint
            dirLen = np.sqrt(direction[0]*direction[0]+direction[1]*direction[1])
            direction /= dirLen # normalized direction between 2 points
            reqUnits = int(np.round(dirLen/unitLength))
            xcoords = np.linspace(startPoint[0], nextPoint[0], num=reqUnits)
            ycoords = np.linspace(startPoint[1], nextPoint[1], num=reqUnits)
            lut[j:(j+reqUnits),0] = xcoords
            lut[j:(j+reqUnits),1] = ycoords
            direction *= unitLength # set the length of the direction vector to contour unit length
            j += reqUnits
        
        lut_res = np.array(lut[0:j,:])
        #idxes = np.linspace(0,j-1, num=j).astype(int)
        #lut_idx = idxes[0:j:int(self.resolution/self.cPoints)]


        
        lut_idx = np.empty(self.cPoints, dtype=int)
        j = 0
        for i in range(0, self.resolution, int(self.resolution/self.cPoints)):
            lut_idx[j] = i
            j += 1
        return lut_res, lut_idx
    
    def export(self, filename):
        with open (filename, 'w') as expfile:
            for i in range(self.nPoi):
                expfile.write(str(self.pointArray[i,0])+","+str(self.pointArray[i,1])+"\n")
            expfile.close()

