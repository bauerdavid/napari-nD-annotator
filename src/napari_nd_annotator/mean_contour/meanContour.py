# from contour import *
from qtpy.QtCore import QThread, Signal
import sys
import numpy as np
import time
import os
from .settings import Settings
from ._reconstruction import reconstruct
from ._contour import initCentroid, delta_d, calcRpsvInterpolation
from ._essentials import magnitude, Contour
from .util import loadContour

class MeanThread(QThread):

    doneSignal = Signal(object)
    clearPlotSignal = Signal()
    updateSignal = Signal(float)
    rpSignal = Signal(object)
    reconSignal = Signal(object)

    def __init__(self, contours, settings=None, weights=None):
        self.settings = settings if settings else Settings()
        self.contours = contours if isinstance(contours[0], Contour)\
            else list(Contour(c.copy(), self.settings.nPoi, self.settings.resMultiplier) for c in contours)
        QThread.__init__(self)
        self.iterations = self.settings.maxIter
        self.weights = weights
    
    def __del__(self):
        self.wait()

    def run(self):

        self.updateSignal.emit(0)

        # settings for the algorithm
        settings = self.settings
        weights = np.ones(len(self.contours)) if self.weights is None else self.weights

        for i in range(len(self.contours)):
            self.contours[i].setStartingPointToLowestY()

        # init centroid at first (take average)
        startCentroid = initCentroid(self.contours, weights)

        # translate every contour
        for i_contour in range(len(self.contours)):
            self.contours[i_contour].lookup -= startCentroid
        
        # weights for interpolation

        properCentroid = startCentroid.copy()
        deltaPrev = np.zeros((1,2))

        # calculate initial mean
        imean = np.zeros((self.contours[0].parameterization.shape[0], 2))
        for i in range(len(self.contours)):
            # TODO weighting (?)
            if not self.contours[i].isClockwise():
                # if the orientation of the polygon is cclockwise, revert it
                self.contours[i].lookup = self.contours[i].lookup[::-1]
            imean += self.contours[i].lookup[self.contours[i].parameterization,:]*weights[i]
        imean /= np.sum(weights)

        # go for the maximum number of iterations (general > maxIter in settings)
        c = 0
        for i in range(self.iterations):
            print("iteration #%d" % i)
            timestamp = time.time()
            regularMean = np.zeros_like(self.contours[0].lookup[self.contours[0].parameterization, :])
            for j in range(len(self.contours)):
                regularMean += self.contours[j].lookup[self.contours[j].parameterization, :]*weights[j]
            regularMean /= np.sum(weights)
            self.contours[1].calcParams()
            # calculate the mean in RPSV space
            q_mean = calcRpsvInterpolation(self.contours, weights)
            # here we initialize the ray lengths for the reconstruction: just take the original averages
            guessRayLengths = np.zeros(self.contours[0].lookup[self.contours[0].parameterization].shape[0])
            for i_contour in range(len(self.contours)):
                contourtmp = self.contours[i_contour].lookup[self.contours[i_contour].parameterization]
                contourlengths = magnitude(contourtmp)
                guessRayLengths += contourlengths * weights[i_contour]
            guessRayLengths /= np.sum(weights)
            guessRayLengths = magnitude(regularMean)

            # lengths of the q space mean
            qraylengths = magnitude(q_mean)
            qraylengths[qraylengths<1e-99] = 1e-99

            # unit direction vectors
            dirs = q_mean/qraylengths.reshape(qraylengths.shape[0], 1) # unit direction of the mean contour points
            timestamp = time.time()
            # do the reconstruction
            r_mean_lengths, costs = reconstruct(q_mean, guessRayLengths.copy(), settings, self.rpSignal)
            # ----------------------------

            # THE mean contour in r space
            r_mean = dirs * r_mean_lengths.reshape(r_mean_lengths.shape[0], 1)

            rsqrts = qraylengths/r_mean_lengths

            # calculate delta_d displacement
            delta = delta_d(self.contours, q_mean, rsqrts)

            # calculate the differences between the current and previous displacements
            deltaDiff = deltaPrev-delta
            deltaDiff = np.sqrt(np.sum(deltaDiff*deltaDiff))

            deltaPrev = delta.copy()

            if deltaDiff<1.:
                print("centroid converged")
                self.updateSignal.emit(100)
                break
                #refCont.lookup -= delta
                #varCont.lookup -= delta
        
            for i_contour in range(len(self.contours)):
                self.contours[i_contour].lookup -= delta
            
            properCentroid += delta[0,:]
            self.updateSignal.emit(100*(i+1)/self.iterations)

        for i_contour in range(len(self.contours)):
            self.contours[i_contour].lookup += properCentroid
        r_mean += properCentroid
        self.doneSignal.emit(r_mean)


def runCommandLine(args, settings):
    contours = []
    for argind in range(1,len(args)-1):
        contours.append(loadContour(args[argind], settings.nPoi, settings.resMultiplier))
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
                stgs = Settings("settings.json")
                runCommandLine(sys.argv, stgs)
        else:
            print("number of arguments not OK. Try calling the script like \"python meanContour.py cont1.csv cont2.csv ... cont_n.csv meanName\"")
            
    else:
        print("not enough args")
