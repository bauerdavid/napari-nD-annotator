import json
from .util import getReconMethod

class Settings:

    def __init__(self, filename=None, max_iterations=20, debug_mode=False, n_points=1000, resolution_multiplier=1000,
                 iteration_multiplier=10, smooth_reparametrization=False, reconstruction_method="Skip reconstruction",
                 gradient_iterations=100, alpha=0.01, lambd=1.):
        if filename is not None:
            with open(filename) as json_file:
                data = json.load(json_file)
            self.exportDict = data

            # general settings
            self.maxIter = (data['general'])['maxIter']
            self.debug = (data['general'])['debugMode']

            # contour settings
            self.nPoi = (data['contours'])['nPoi']
            self.resMultiplier = (data['contours'])['resMultiplier']

            # reparameterization settings
            self.iterations = self.nPoi*(data['reparameterization'])['iterationMultiplier']
            self.smoothParam = (data['reparameterization'])['smoothReparam']

            # reconstrutcion settings
            self.reconText = (data['reconstruction'])['reconMethod']
            self.reconMethod = getReconMethod((data['reconstruction'])['reconMethod'])
            self.gradientIterations = (data['reconstruction'])['iterations']
            self.alpha = (data['reconstruction'])['alpha']
            self.lambd = (data['reconstruction'])['lambda']

            # export settings
            self.exportCsv = (data['export'])['exportCsv']
            self.exportName = (data['export'])['exportName']
        else:
            # general settings
            self.maxIter = max_iterations
            self.debug = debug_mode

            # contour settings
            self.nPoi = n_points
            self.resMultiplier = resolution_multiplier

            # reparameterization settings
            self.iterations = self.nPoi * iteration_multiplier
            self.smoothParam = smooth_reparametrization

            # reconstrutcion settings
            self.reconText = reconstruction_method
            self.reconMethod = getReconMethod(reconstruction_method)
            self.gradientIterations = gradient_iterations
            self.alpha = alpha
            self.lambd = lambd

    def update(self, category, name, value):
        print("Updating initial entry in category "+category+", name "+name+": "+str((self.exportDict[category])[name])+"...")
        (self.exportDict[category])[name] = value
        print("Updated entry in category "+category+", name "+name+": "+str((self.exportDict[category])[name])+"!")
    
    # update the settings variables according to the exportDict
    def updateVariables(self):
        data = self.exportDict
        
        # general settings
        self.maxIter = (data['general'])['maxIter']
        self.debug = (data['general'])['debugMode']

        # contour settings
        self.nPoi = (data['contours'])['nPoi']
        self.resMultiplier = (data['contours'])['resMultiplier']

        # reparameterization settings
        self.iterations = self.nPoi*(data['reparameterization'])['iterationMultiplier']
        self.smoothParam = (data['reparameterization'])['smoothReparam']

        # reconstrutcion settings
        self.reconText = (data['reconstruction'])['reconMethod']
        self.reconMethod = getReconMethod((data['reconstruction'])['reconMethod'])
        self.gradientIterations = (data['reconstruction'])['iterations']
        self.alpha = (data['reconstruction'])['alpha']
        self.lambd = (data['reconstruction'])['lambda']

        # export settings
        self.exportCsv = (data['export'])['exportCsv']
        self.exportName = (data['export'])['exportName']
