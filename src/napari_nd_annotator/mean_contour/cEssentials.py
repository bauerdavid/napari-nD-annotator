import numpy as np
import matplotlib.pyplot as plt
import math
import time
from enum import Enum
from scipy.interpolate import splprep, splev
class ReconstructionMethods(Enum):
    NEWTON = 1
    GRADIENT_DESCENT = 2
    CG = 3
    SKIP = 4
    JOZSI_GRADIENT = 5


def getCoefficientsForAccuracyLevel(aLevel, order):
    switcher1 = {
        1: [-1/2, 0, 1/2],
        2: [1/12, -2/3, 0, 2/3, -1/12],
        3: [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60],
        4: [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]
    }
    switcher2 = {
        1: [1, -2, 1],
        2: [-1/12, 4/3, -5/2, 4/3, -1/12],
        3: [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90],
        4: [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]
    }

    if order==1:
        return switcher1.get(aLevel)
    if order==2:
        return switcher2.get(aLevel)
    

# derivative approximation
def dt(points, order):

    pNext = np.roll(points, -1, axis=0)
    pPrev = np.roll(points, 1, axis=0)

    pNext2 = np.roll(points, -2, axis=0)
    pPrev2 = np.roll(points, 2, axis=0)

    pNext3 = np.roll(points, -3, axis=0)
    pPrev3 = np.roll(points, 3, axis=0)

    pNext4 = np.roll(points, -4, axis=0)
    pPrev4 = np.roll(points, 4, axis=0)
    #d = points-pPrev
    #e = pNext-points
    #d_abs = magnitude(d).reshape(d.shape[0],1)
    #e_abs = magnitude(e).reshape(e.shape[0],1)

    if order==1:
        prevFactors = (1/280)*pPrev4-(4/105)*pPrev3+(1/5)*pPrev2-(4/5)*pPrev
        nextFactors = -(1/280)*pNext4+(4/105)*pNext3-(1/5)*pNext2+(4/5)*pNext
        return prevFactors+nextFactors
        #retval = ((d_abs*e_abs)/(d_abs+e_abs))*(d/d_abs**2 + e/e_abs**2)
        #return retval/magnitude(retval).reshape(d.shape[0],1)
        #return (pNext-pPrev)/2
        #return (1/12)*pPrev2-(2/3)*pPrev+(2/3)*pNext-(1/12)*pNext2

    if order==2:
        prevFactors = (-1/560)*pPrev4+(8/315)*pPrev3-(1/5)*pPrev2+(8/5)*pPrev
        currFactor = (-205/72)*points
        nextFactors = (-1/560)*pNext4+(8/315)*pNext3-(1/5)*pNext2+(8/5)*pNext
        return prevFactors+currFactor+nextFactors
        #retval = (2/(d_abs+e_abs))*(e/e_abs - d/d_abs)
        #return retval/magnitude(retval).reshape(d.shape[0],1)
        #return pNext+pPrev-2*points
        #return (-1/12)*pPrev2+(4/3)*pPrev-(5/2)*points+(4/3)*pNext-(1/12)*pNext2


# magnitude of a single vector or a set of vectors
def magnitude(points):
    if len(points.shape)>1:
        return np.sqrt(points[:,0]*points[:,0]+points[:,1]*points[:,1])
    else:
        return np.sqrt(points[0]*points[0]+points[1]*points[1])
        
# inner product of two curves
def innerProduct(curve1,curve2):
    return np.sum(curve1*curve2, axis=1)
    # return curve1[:,0]*curve2[:,0]+curve1[:,1]*curve2[:,1]

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
        #print("unit length: "+str(unitLength))
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
            #print("dirlen: "+str(dirLen))
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
        #print(xcoords)
        #print(j)
        return lut_res, lut_idx
    
    def export(self, filename):
        with open (filename, 'w') as expfile:
            for i in range(self.nPoi):
                expfile.write(str(self.pointArray[i,0])+","+str(self.pointArray[i,1])+"\n")
            expfile.close()

