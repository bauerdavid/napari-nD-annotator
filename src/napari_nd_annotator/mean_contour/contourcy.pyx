# from cEssentials import *
from ._essentials import magnitude, innerProduct
cimport numpy as np
cimport cython
from cython.parallel cimport prange
import numpy as np
import time

# determine the initial centroid, just sum up the coordinates and take their average
def initCentroid(contours, weights = None):
    r = np.zeros((1,2))
    weights = np.ones(len(contours)) if weights is None else weights
    for i in range(len(contours)):
        lut = contours[i].lookup[contours[i].parameterization]
        r += np.sum(lut, axis=0) * weights[i]
    r /= (np.sum(weights)*lut.shape[0])
    return r

def isDifferenceSmallEnough(currVelo, prevVelo):
    if prevVelo is not None:
        scurr = magnitude(currVelo)
        sprev = magnitude(prevVelo)
        diffSum = np.sum(np.abs(sprev-scurr), axis=0)/scurr.shape[0]
        if diffSum < 0.00001:
            print("reparam converged: "+str(diffSum))
            return True
        else:
            return False
    else: return False

def isMethodStuck(prevMean, energies):
    if np.abs(prevMean-np.mean(energies))<0.01:
        return True
    else:
        return False

def reparameterizeContours(contours, Settings, plotSignal=None, debug=False):
    refCont = contours[0]
    numIterations = Settings.iterations
    maxgamma = 0.24
    # maxgamma = 0.1
    refCont.calcParams()
    refRepr = refCont.getRPSV()
    nPoints = Settings.nPoi


    #fig = plt.figure()
    #camera = Camera(fig)
    sumtime = 0
    sumsmooth = 0
    g = contours[1].parameterization
    for cIter in range(1,len(contours)):
        varCont = contours[cIter]
        #varCont.parameterization = g
        stopCriterion = False
        prevDerivatives = None
        min_resolution = varCont.resMultiplier
        energies = []
        prevEnergies = -1
        costs = []
        for i in range(numIterations):
            if stopCriterion is True:
                break

            '''if i%2==0:
                varCont = contours[1]
                refCont = contours[0]
            else:
                varCont = contours[0]
                refCont = contours[1]'''

            start = time.time()
            varCont.calcParams()
            end = time.time()
            sumtime += (end-start)

            # gradient descent equation as defined in (16):
            dgamma = 0.5*(refCont.lookup[refCont.parameterization,0]*varCont.lookup[varCont.parameterization,0]+refCont.lookup[refCont.parameterization,1]*varCont.lookup[varCont.parameterization,1]+0)*(refCont.christoffel-varCont.christoffel)
            dgamma += (refCont.derivatives[:,0]*varCont.lookup[varCont.parameterization,0]+refCont.derivatives[:,1]*varCont.lookup[varCont.parameterization,1])
            dgamma -= (refCont.lookup[refCont.parameterization,0]*varCont.derivatives[:,0]+refCont.lookup[refCont.parameterization,1]*varCont.derivatives[:,1])
            dgamma *= -1
            
            maxgammaabs = np.max(np.abs(dgamma))

            if maxgammaabs<1e-99:
                maxgammaabs = 1e-99

            if maxgammaabs>maxgamma:
                dgamma /= maxgammaabs
                dgamma *= maxgamma
            
            

            #distances = np.sum(np.abs(np.sqrt(refCont.srv)-np.sqrt(varCont.srv)))
        
            varRepr = varCont.getRPSV()
            
            qdistances = np.sum(np.sqrt(innerProduct(refRepr-varRepr, refRepr-varRepr)))
            costs.append(qdistances)

            # if the optimitzer gets stuck at a point, detect it
            if (i+1)%300 == 0:
                print("energy_rp #"+str(i)+": "+str(qdistances))
                if isMethodStuck(prevEnergies, energies):
                    print("reparam stuck")
                    stopCriterion = True
                prevEnergies = np.mean(energies)
                energies.clear()
            else:
                energies.append(qdistances)


            # update with dGamma

            idx_diff = varCont.getIdxDiff(dgamma)
            g = (varCont.parameterization+idx_diff*dgamma+0.5).astype(int)
            varCont.parameterization = g

            # when we got our parameterization, make sure that we remain in range of the lookup table
            varCont.getInRangeParameterization()

            # smoothing to ensure that the order of points does not get mixed up
            startsmooth = time.time()
            varCont.smoothParameterization()
            endsmooth = time.time()
            sumsmooth += (endsmooth-startsmooth)

            #varCont.getInRangeParameterization()

            g = varCont.parameterization

            meantest = (refCont.lookup[refCont.parameterization,:]+varCont.lookup[varCont.parameterization,:])/2

            if Settings.debug is True:
                if (i%200==0):
                    cList = []
                    cList.append(refCont.lookup[refCont.parameterization[0:nPoints:15],:])
                    cList.append(varCont.lookup[varCont.parameterization[0:nPoints:15],:])
                    cList.append(meantest[0:nPoints:15,:])
                    plotSignal.emit(cList)



            if isDifferenceSmallEnough(varCont.derivatives, prevDerivatives):
                stopCriterion = True

            #avg = (refCont.lookup[refCont.parameterization]+varCont.lookup[varCont.parameterization])/2

            prevDerivatives = varCont.derivatives.copy()

        return costs


    #print("sum time spent on calcparams: "+str(sumtime))
    #print("sum time spent on smoothing: "+str(sumsmooth))

# calculate the weighted mean of contours in the rpsv space
def calcRpsvInterpolation(contours, weights):
    interp = np.zeros(contours[0].getRPSV().shape)

    # take the average in q-space: interp
    for i in range(len(contours)):
        r = contours[i].getRPSV()
        r *= weights[i]
        interp += r
    interp /= np.sum(weights)
    return interp

def calcMean(contours):
    N = len(contours)
    mean = contours[0].copy()
    for i in range(1,len(contours)):
        mean += contours[i]
    mean /= N
    return mean

# calculate centroid displacement
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef delta_d(contours, np.ndarray[np.double_t, ndim=2] q_mean, np.ndarray[np.double_t, ndim=1] rsqrts):
    cdef double energy = 0
    cdef double denominator = 0
    cdef np.ndarray[np.double_t, ndim=2] numerator = np.zeros((1,2))
    cdef np.ndarray[np.double_t, ndim=1] targetSrv
    cdef np.ndarray[np.double_t, ndim=2] targetRepr
    cdef double denominatori
    cdef np.ndarray[np.double_t, ndim=2] numeratori
    cdef int i, j

    for i in range(len(contours)):
        targetCont = contours[i]
        
        targetSrv = targetCont.getSRV()
        targetRepr = targetCont.getRPSV()
        
        denominatori = 0
        numeratori = np.zeros((1,2))

        for j in prange(targetRepr.shape[0], nogil=True):
            targetRepr[j, 0] -= q_mean[j, 0]
            targetRepr[j, 1] -= q_mean[j, 1]
            energy += targetRepr[j,0]*targetRepr[j,0]+targetRepr[j,1]*targetRepr[j,1]
            targetRepr[j, 0] *= targetSrv[j]
            targetRepr[j, 1] *= targetSrv[j]
            numerator[0, 0] += targetRepr[j, 0]
            numerator[0, 1] += targetRepr[j, 1]
            numeratori[0, 0] += targetRepr[j, 0]
            numeratori[0, 1] += targetRepr[j, 1]
            denominatori += targetSrv[j]*targetSrv[j]
            targetSrv[j] -= rsqrts[j]
            denominator += targetSrv[j]*targetSrv[j]

        targetCont.centroid[0,0] = numeratori[0,0]/denominatori
        targetCont.centroid[0,1] = numeratori[0,1]/denominatori

    senergy = np.sqrt(energy)

    if denominator>1e-99:
        delta = numerator
        delta /= denominator

    return delta
