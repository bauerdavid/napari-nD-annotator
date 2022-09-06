import numpy as np
from ._essentials import ReconstructionMethods, Contour
import matplotlib.pyplot as plt

# loads a contour from file contName with nPoi points and a resolution of nPoi*resMultiplier
def loadContour(contName, nPoi, resMultiplier):
    pts = np.genfromtxt(contName, delimiter=",")
    res = Contour(pts.copy(), nPoi, resMultiplier)
    return res

# plots a 1-variate function
def plotFunction(x, fx, title=""):
    plotfig = plt.figure()
    plt.title(title)
    plt.plot(x,fx)
    return plotfig


def plotContours(contours,labels=None,colors=None,title=""):
    plotfig = plt.figure()
    plt.title(title)
    for i in range(len(contours)):
        curr = contours[i]
        clabel = ""
        ccolor = "black"
        if labels:
            clabel = labels[i]
        if colors:
            ccolor = colors[i]
        plt.plot(curr[:,1],-1*curr[:,0], label=clabel, color=ccolor)
    plt.legend()
    return plotfig

def getReconMethod(reconText):
    switcher = {
        'Newton': ReconstructionMethods.NEWTON,
        'Gradient descent': ReconstructionMethods.GRADIENT_DESCENT,
        'Conjugate gradient': ReconstructionMethods.CG,
        'Skip reconstruction': ReconstructionMethods.SKIP,
        'Jozsi gradient': ReconstructionMethods.JOZSI_GRADIENT
    }
    return switcher.get(reconText)