import scipy.interpolate as interp
class InterpHelper:
    def __init__(self):
        pass
    def setInterpolator(self,xpoints,ypoints):
        self.interpolator = interp.interp1d(x=xpoints,y=ypoints,kind='quadratic')
