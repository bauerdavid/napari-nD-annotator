import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,5,num=6)
y = [0,1,4,9,16,25]
finterp = interp.interp1d(x=x,y=y,kind='quadratic')


xnew = np.linspace(0,5,num=100)
plt.plot(xnew,finterp(xnew))
plt.show()