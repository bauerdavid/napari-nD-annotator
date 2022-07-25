import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs': np.get_include()})
# from contour import *
from contourcy import initCentroid, calcRpsvInterpolation, delta_d
from PyQt5.QtCore import QThread, pyqtSignal
import sys
import numpy as np
import time
import os
import settings
import reconstructioncy as reconstruction
# import reconstruction
import cEssentialscy as cEssentials
# import cEssentials
# import reconstruction
import util

class MeanThread(QThread):

    doneSignal = pyqtSignal(object)
    clearPlotSignal = pyqtSignal()
    updateSignal = pyqtSignal(float)
    rpSignal = pyqtSignal(object)
    reconSignal = pyqtSignal(object)

    def __init__(self, contours, settings):
        self.contours = contours if isinstance(contours[0], cEssentials.Contour)\
            else list(cEssentials.Contour(c.copy(), settings.nPoi, settings.resMultiplier) for c in contours)
        QThread.__init__(self)
        self.settings = settings
        self.iterations = settings.maxIter
    
    def __del__(self):
        self.wait()

    def run(self):

        self.updateSignal.emit(0)

        # settings for the algorithm
        settings = self.settings

        for i in range(len(self.contours)):
            self.contours[i].setStartingPointToLowestY()

        # init centroid at first (take average)
        startCentroid = initCentroid(self.contours)

        # translate every contour
        for i_contour in range(len(self.contours)):
            self.contours[i_contour].lookup -= startCentroid
        
        # weights for interpolation
        weights = np.ones(len(self.contours))

        properCentroid = startCentroid.copy()
        deltaPrev = np.zeros((1,2))
        deltaSum = np.zeros((1,2))

        # calculate initial mean
        imean = np.zeros((self.contours[0].parameterization.shape[0], 2))
        for i in range(len(self.contours)):
            if not self.contours[i].isClockwise():
                # if the orientation of the polygon is cclockwise, revert it
                self.contours[i].lookup = self.contours[i].lookup[::-1]
            imean += self.contours[i].lookup[self.contours[i].parameterization,:]
        imean /= len(self.contours)

        # go for the maximum number of iterations (general > maxIter in settings)
        c = 0
        for i in range(self.iterations):

            print("iteration#"+str(i))
            iternum = i

            regularMean = (self.contours[0].lookup[self.contours[0].parameterization,:]+self.contours[1].lookup[self.contours[1].parameterization,:])/2
            start = time.time()
            self.contours[1].calcParams()
            # calculate the mean in RPSV space
            q_mean = calcRpsvInterpolation(self.contours, weights)
            # here we initialize the ray lengths for the reconstruction: just take the original averages
            guessRayLengths = np.zeros(self.contours[0].lookup[self.contours[0].parameterization].shape[0])
            for i_contour in range(len(self.contours)):
                contourtmp = self.contours[i_contour].lookup[self.contours[i_contour].parameterization].copy()
                contourlengths = cEssentials.magnitude(contourtmp)
                guessRayLengths += contourlengths
            guessRayLengths /= np.sum(weights, axis=0)
            guessRayLengths = cEssentials.magnitude(regularMean)

            # lengths of the q space mean
            qraylengths = cEssentials.magnitude(q_mean)
            qraylengths[qraylengths<1e-99] = 1e-99

            # unit direction vectors
            dirs = q_mean/qraylengths.reshape(qraylengths.shape[0],1) # unit direction of the mean contour points

            # do the reconstruction
            r_mean_lengths, costs = reconstruction.reconstruct(q_mean, guessRayLengths.copy(), settings, self.rpSignal)
            # ----------------------------

            # THE mean contour in r space
            r_mean = dirs*r_mean_lengths.reshape(r_mean_lengths.shape[0],1)

            rsqrts = qraylengths/r_mean_lengths 
            timestamp = time.time_ns()

            # calculate delta_d displacement
            delta = delta_d(self.contours, q_mean, rsqrts)
            print("delta:", time.time_ns()-timestamp)

            # calculate the differences between the current and previous displacements
            deltaDiff = np.sqrt((deltaPrev[0,0]-delta[0,0])*(deltaPrev[0,0]-delta[0,0])+(deltaPrev[0,1]-delta[0,1])*(deltaPrev[0,1]-delta[0,1]))

            deltaPrev = delta.copy()

            #original mean
            om = np.ndarray(r_mean.shape)
            om[:,0] = dirs[:,0]*guessRayLengths
            om[:,1] = dirs[:,1]*guessRayLengths
            omVelo = cEssentials.magnitude(cEssentials.dt(om,1))
            omSrv = np.sqrt(omVelo)
            om[:,0]*=omSrv
            om[:,1]*=omSrv

            if 0.5*deltaDiff<0.5:
                print("centroid converged")
                self.updateSignal.emit(100)
                break
                #refCont.lookup -= delta
                #varCont.lookup -= delta
        
            for i_contour in range(len(self.contours)):
                self.contours[i_contour].lookup -= delta
                
            deltaSum += delta

            end = time.time()
            print("time spent on reconstruction: "+str(end-start))
            
            properCentroid += delta[0,:]
            self.updateSignal.emit(100*(iternum+1)/self.iterations)

        for i_contour in range(len(self.contours)):
            self.contours[i_contour].lookup += properCentroid
        r_mean += properCentroid
        #self.clearPlotSignal.emit()

        self.doneSignal.emit(r_mean)


def runCommandLine(args, settings):
    contours = []
    for argind in range(1,len(args)-1):
        contours.append(util.loadContour(args[argind], settings.nPoi, settings.resMultiplier))
    settings.update("export", "exportName", args[len(args)-1])
    settings.updateVariables()
    meanThread = MeanThread(contours.copy(), settings)
    meanThread.start()


if __name__ == '__main__':
    if len(sys.argv)>1:
        print("running in command line mode.")
        if len(sys.argv) >= 4:
            print("number of arguments OK. Trying to load the folllowing contour files: ")
            for i in range(1,len(sys.argv)-1):
                print(sys.argv[i])
            if not os.path.exists("settings.json"):
                print("no settings.json found. Aborting...")
            else:
                settings = settings.Settings("settings.json")
                runCommandLine(sys.argv, settings)
        else:
            print("number of arguments not OK. Try calling the script like \"python meanContour.py cont1.csv cont2.csv ... cont_n.csv meanName\"")
            
    else:
        print("not enough args")
