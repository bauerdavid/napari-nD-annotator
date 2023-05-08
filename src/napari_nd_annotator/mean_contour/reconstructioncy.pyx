# cython: language_level=3, language = c++
cimport cython
from cython.parallel cimport prange
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport fabs, sqrt, cbrt
from ._essentials import dt, magnitude, innerProduct, ReconstructionMethods
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
from scipy.optimize import minimize
from scipy.integrate import solve_bvp
from .rk import rk4
from .interpHelper import *
import time
import matplotlib.pyplot as plt
Q_interp = InterpHelper()
Qdot_interp = InterpHelper()
theta_interp = InterpHelper()
thetadot_interp = InterpHelper()
thetaddot_interp = InterpHelper()


# function that is called from the main executor
def reconstruct(np.ndarray[np.double_t, ndim=2] q_mean, np.ndarray[np.double_t, ndim=1] guessRayLengths, settings, signal):
    if settings.reconMethod is ReconstructionMethods.NEWTON:
        return reconstruct_newton(q_mean, guessRayLengths, settings.debug, signal)
    if settings.reconMethod is ReconstructionMethods.GRADIENT_DESCENT:
        return gDescent(q_mean,guessRayLengths, settings)
    if settings.reconMethod is ReconstructionMethods.CG:
        return reconstruct_cgradient(q_mean, guessRayLengths, settings, signal)
    if settings.reconMethod is ReconstructionMethods.SKIP:
        return reconstruct_iterate_implicit(q_mean,guessRayLengths)
    if settings.reconMethod is ReconstructionMethods.JOZSI_GRADIENT:
        return reconstruct_gradient(q_mean, guessRayLengths, 100, settings.debug, signal)
    raise NotImplementedError("Method %s not implemented" % settings.reconMethod)


# cost function for gradient descent, x: |r|, qraylengths: |q|, dirs are the unit direction vectors for q and r.
def cost(x, qraylengths, dirs):
    rguess = dirs*x.reshape(x.shape[0],1)
    rdot = dt(rguess,1)
    rdot_abs = magnitude(rdot)
    srv = np.sqrt(rdot_abs)
    return (qraylengths-x*srv)**2

# derivative of the cost function with respect to x (|r|)
def dcost(x, qraylengths, dirs):
    rguess = dirs*x.reshape(x.shape[0],1) # r
    q = dirs*qraylengths.reshape(qraylengths.shape[0],1) # q

    rdot = dt(rguess,1)
    rdotdot = dt(rguess,2)
    rdot_abs = magnitude(rdot)
    qdot = dt(q,1)
    srv = np.sqrt(rdot_abs)

    nominator = innerProduct(q,qdot)/qraylengths - ( (innerProduct(rguess,rdot)/x)*srv + (x/(2*srv))*(innerProduct(rdot,rdotdot)/rdot_abs) )
    denominator = innerProduct(rguess,rdot)

    gradients = x*(nominator/denominator)
    gradients *= 2*(qraylengths-x*srv)

    return gradients

# gradient descent algorithm
def gDescent(q, guessRayLengths, settings):
    qraylengths = np.sqrt(innerProduct(q,q))
    dirs = q/qraylengths.reshape(qraylengths.shape[0],1)
    alpha = settings.alpha
    iters = settings.gradientIterations
    maxgrad = 1.0
    costs = []
    for i in range(iters):
        # calculate cost
        cost_sum = np.sum(cost(guessRayLengths,qraylengths,dirs))
        print("cost = "+str(cost_sum))
        costs.append(cost_sum)

        # calculate the derivative of the cost wrt |r|
        gradients = dcost(guessRayLengths, qraylengths, dirs)

        # make sure that no gradient is too large, rescale them so that the biggest gradient is of size maxgrad at most.
        curr_max = np.max(np.abs(gradients))
        if (curr_max>maxgrad):
            gradients /= curr_max
            gradients *= maxgrad
        
        # gradient descent update rule
        guessRayLengths = guessRayLengths-alpha*gradients
        #guessNext = np.roll(guessRayLengths,-1,axis=0)
        #guessPrev = np.roll(guessRayLengths,1,axis=0)
        #guessRayLengths = 0.1*(8*guessRayLengths+guessNext+guessPrev)

    return guessRayLengths, costs


# gradient descent reconstruction of a q-space curve with an initial guess: will return the length of each ray
def reconstruct_gradient(q_mean, guessRayLengths, multiplier, debug, plotSignal):
    qraylengths = np.sqrt(q_mean[:,0]*q_mean[:,0] + q_mean[:,1]*q_mean[:,1])
    qraylengths[qraylengths<1e-99] = 1e-99
    dirs = q_mean.copy()

    dirs[:,0] /= qraylengths # direction of the mean contour points
    dirs[:,1] /= qraylengths

    startRepr = dirs.copy()
    #if np.abs(q-0.5) < 0.01:
    startRepr[:,0]*=qraylengths
    startRepr[:,1]*=qraylengths
    
    nIter = 10*multiplier
    costs = []
    for i in range(nIter):
        rguess = dirs*guessRayLengths.reshape(guessRayLengths.shape[0],1)
        rdot = dt(rguess,1)
        rdot_abs = magnitude(rdot)
        srv = np.sqrt(rdot_abs)

        energy = qraylengths-guessRayLengths*srv
        energy *= energy
        energy = np.sum(energy, axis=0)
        costs.append(energy)

        rayDiff = qraylengths-(guessRayLengths*srv)

        maxDiff = np.max(np.abs(rayDiff))

        rayDiff *= float(1/maxDiff)

        rayLengths = guessRayLengths + 0.1*rayDiff

        rayNext = np.roll(rayLengths, -1, axis=0)
        rayPrev = np.roll(rayLengths, 1, axis=0)

        #guessRayLengths = rayLengths

        guessRayLengths = 0.1*(8*rayLengths + rayPrev + rayNext)
        
        # display the evolution
        if debug is True and i%10==0:
            cList = []
            restmp = dirs.copy()
            restmp[:,0] *=guessRayLengths*srv
            restmp[:,1] *=guessRayLengths*srv
            cList.append(q_mean[0:q_mean.shape[0]:15,:])
            cList.append(restmp[0:restmp.shape[0]:15,:])
            plotSignal.emit(cList)
            time.sleep(0.1)
        
    return rayLengths, costs

# Newton-Raphson method
def reconstruct_newton(q_mean, guessRayLengths, debug, plotSignal):

    # to avoid a too large condition number
    # guessRayLengths

    bandlength = guessRayLengths.shape[0]-1
    x = guessRayLengths.copy()
    x_prev = guessRayLengths.copy()
    A = np.ndarray((guessRayLengths.shape[0], guessRayLengths.shape[0]), dtype=float)
    b = np.array(guessRayLengths.shape[0], dtype=float)
    upperband = np.array(bandlength, dtype=float)
    lowerband = np.array(bandlength, dtype=float)

    qraylengths = np.sqrt(q_mean[:,0]*q_mean[:,0] + q_mean[:,1]*q_mean[:,1])
    qraylengths[qraylengths<1e-99] = 1e-99
    dirs = q_mean.copy()

    dirs[:,0] /= qraylengths # unit direction of the mean contour points
    dirs[:,1] /= qraylengths

    lrate = 0.01#/guessRayLengths.shape[0]
    nIter = 10*10
    stopCriterion = False
    xmaxglob = 0
    for i in range(nIter):
        if stopCriterion is True:
            break
        dirNext = np.roll(dirs, -1, axis=0)
        dirPrev = np.roll(dirs, 1, axis=0)
        dirNextUnit = np.roll(dirs, -1, axis=0)
        dirPrevUnit = np.roll(dirs, 1, axis=0)
        guessNext = np.roll(guessRayLengths, -1)
        guessPrev = np.roll(guessRayLengths, 1)
        dirNext[:,0]*=guessNext
        dirNext[:,1]*=guessNext
        dirPrev[:,0]*=guessPrev
        dirPrev[:,1]*=guessPrev
        velo = (dirNext-dirPrev)*0.5
        norm_velo = np.sqrt(velo[:,0]*velo[:,0]+velo[:,1]*velo[:,1])
        e = np.ndarray(velo.shape)
        e[:,0] = velo[:,0]/norm_velo
        e[:,1] = velo[:,1]/norm_velo
        velo2 = velo[:,0]*velo[:,0]+velo[:,1]*velo[:,1]
        srv = np.sqrt(np.sqrt(velo2))

        energy = qraylengths-guessRayLengths*np.sqrt(norm_velo)
        energy *= energy
        energy = np.sqrt(np.sum(energy, axis=0))
 
        dirnext_velo_prod = dirNextUnit[:,0]*e[:,0]+dirNextUnit[:,1]*e[:,1]
        dirprev_velo_prod = dirPrevUnit[:,0]*e[:,0]+dirPrevUnit[:,1]*e[:,1]

        upperband = 0.25 * (guessRayLengths[:bandlength-1]/(norm_velo[:bandlength-1])) * dirnext_velo_prod[bandlength-1]
        lowerband = -0.25 * (guessRayLengths[:bandlength-1]/(norm_velo[:bandlength-1])) * dirprev_velo_prod[bandlength-1]
        #upperband = 0.25 * guessRayLengths[:bandlength-1] * (dirnext_velo_prod[:bandlength-1]/velo2[:bandlength-1])
        #lowerband = -0.25 * guessRayLengths[:bandlength-1] * (dirprev_velo_prod[:bandlength-1]/velo2[:bandlength-1])
        A = np.ndarray((guessRayLengths.shape[0], guessRayLengths.shape[0]))
        
        A[:,:] = 0.0
        '''
        print('-----------------------')
        print("min abs upperband: "+str(np.min(np.abs(upperband))))
        print("min abs lowerband: "+str(np.min(np.abs(lowerband))))
        print("max abs upperband: "+str(np.max(np.abs(upperband))))
        print("max abs lowerband: "+str(np.max(np.abs(lowerband))))
        print('-----------------------')
        '''
        np.fill_diagonal(A, 1.0)
        np.fill_diagonal(A[:,1:], upperband)
        np.fill_diagonal(A[1:,:], lowerband)
        A[0,bandlength] = -0.25 * (guessRayLengths[1]/(norm_velo[1])) * dirprev_velo_prod[1]
        A[bandlength,0] = 0.25 * (guessRayLengths[bandlength]/(norm_velo[bandlength])) * dirnext_velo_prod[bandlength]
        b = ( qraylengths / np.sqrt(norm_velo) ) + 0.5*guessRayLengths
        if i==0:
            condnum = np.linalg.cond(A)
            print('-----------------------')
            print("condition number of matrix: "+str(condnum))
            print("min abs upperband: "+str(np.min(np.abs(upperband)))+", min lower "+str(np.min(np.abs(lowerband)))+", max upper "+str(np.max(np.abs(upperband)))+", max lower "+str(np.max(np.abs(lowerband))))
            print("min denominator in A: "+str(np.min(norm_velo))+", max denominator in A: "+str(np.max(norm_velo)))
            print('-----------------------')
        A = csc_matrix(A)

        
        u, s, v = svds(A)
        sm = np.zeros((s.shape[0], s.shape[0]))
        np.fill_diagonal(sm, s)
        x = np.matrix.transpose(v) @ np.linalg.inv(sm) @ np.matrix.transpose(u) @ b
        
        #x = np.linalg.solve(A,b)
        
        # smoothing
        #xn = np.roll(x, -1)
        #x2n = np.roll(x,-2)
        #xp = np.roll(x, 1)
        #x2p = np.roll(x,2)
        #x = (2*xn+x2n+2*xp+x2p+4*x)/10
        #----------------
        
        if np.max(np.abs(x)) > xmaxglob:
            xmaxglob = np.max(np.abs(x))


        x = (1-lrate)*x_prev + lrate*x

        guessRayLengths = x.copy()

        if debug is True and i%5==0:
            cList = []
            restmp = dirs.copy()
            dirNext = np.roll(dirs, -1, axis=0)
            dirPrev = np.roll(dirs, 1, axis=0)
            guessNext = np.roll(guessRayLengths, -1)
            guessPrev = np.roll(guessRayLengths, 1)
            dirNext[:,0]*=guessNext
            dirNext[:,1]*=guessNext
            dirPrev[:,0]*=guessPrev
            dirPrev[:,1]*=guessPrev
            velo = (dirNext-dirPrev)*0.5
            norm_velo = np.sqrt(velo[:,0]*velo[:,0]+velo[:,1]*velo[:,1])
            srv = np.sqrt(norm_velo)
            restmp[:,0] *=guessRayLengths#*srv
            restmp[:,1] *=guessRayLengths#*srv
            cList.append(q_mean[0:q_mean.shape[0]:15,:])
            cList.append(restmp[0:restmp.shape[0]:15,:])
            plotSignal.emit(cList)
            time.sleep(0.1)


        xdiff = x-x_prev
        absxdiffmean = np.mean(np.sqrt(xdiff*xdiff))
        if absxdiffmean < 0.001:
            stopCriterion = True
        x_prev = x.copy()
    print("max x during opt:"+str(xmaxglob))
    print("x max after:"+str(np.max(np.abs(guessRayLengths))))
    return guessRayLengths

# cost function of the reconstruction: used at the conjugate gradient method as this is a sum
def energyfunc(x, qraylengths, dirs, lambd):

    rguess = dirs*x.reshape(x.shape[0],1)
    velo = magnitude(dt(rguess,1))
    curvature = magnitude(dt(rguess,2))
    kappanom = np.abs(dt(rguess[:,0],1)*dt(rguess[:,1],2)-dt(rguess[:,1],1)*dt(rguess[:,0],2))
    kappadenom = (dt(rguess[:,0],1)**2 + dt(rguess[:,1],1)**2)**(3/2)
    kappa = kappanom/kappadenom
    energy = qraylengths-x*np.sqrt(velo)
    energy *= energy

    energy = lambd*energy+(1-lambd)*kappa

    return np.sum(energy, axis=0)


# conjugate gradient reconstruction
def reconstruct_cgradient(q_mean, guessRayLengths, settings, plotSignal=None):
    #guessRayLengths = reconstruct_gradient(q_mean, guessRayLengths.copy(), debug, plotSignal)
    
    qraylengths = np.sqrt(q_mean[:,0]*q_mean[:,0] + q_mean[:,1]*q_mean[:,1])
    qraylengths[qraylengths<1e-99] = 1e-99

    dirs = q_mean.copy()

    dirs /= qraylengths.reshape(qraylengths.shape[0], 1) # direction of the mean contour points

    lambd = settings.lambd

    opres = minimize(energyfunc, guessRayLengths, args=(qraylengths, dirs, lambd), method='CG',options={'maxiter': 1000, 'eps': 1.4901161193847656e-8})
    print("final cost: "+str(opres.fun))
    x = opres.x
    #x = optimize_gradient(guessRayLengths,qraylengths,dirs)

    return x, []

# for R-K reconstruction
def getPotentialZeroPoints(q, guessRayLengths):
    q_abs = magnitude(q)
    q_abs_dot = dt(q_abs,1)
    q_x = q[:,0]
    q_y = q[:,1]
    q_x_dot = dt(q_x, 1)
    q_y_dot = dt(q_y, 1)
    q_x_ddot = dt(q_x, 2)
    q_y_ddot = dt(q_y, 2)
    theta = np.arctan2(q_y,q_x)
    theta_dot = dt(theta, 1)#(q_y_dot*q_x - q_y*q_x_dot)/(q_x*q_x+q_y*q_y)
    theta_ddot = dt(theta, 2)#(q_x*q_y_ddot - q_y*q_x_ddot)/(q_x*q_x+q_y*q_y) -2*((q_x*q_y_dot-q_y*q_x_dot)*(q_y*q_y_dot+q_x*q_x_dot))/(q_x*q_x+q_y*q_y)**2

    diffs = 2*q_abs_dot*theta_dot - q_abs*theta_ddot
    pointnum = q_abs.shape[0]
    candidate_num = diffs[diffs<1].shape[0]
    print(diffs)

    idxes = np.argwhere(theta_dot<-2)
    theta_dot[theta_dot<-0.5] = 0
    
    indices = np.argsort(diffs)
    f = plt.figure()
    plt.plot(q[:,1], -1*q[:,0])
    plt.scatter(q[idxes,1], -1*q[idxes,0], label="theta_dot<-2", color="r")
    plt.legend()
    f.savefig("candidates.png", dpi=300)

    r_abs_dot = dt(guessRayLengths,1)
    r_abs_ddot = dt(guessRayLengths,2)
    t = np.linspace(0,len(guessRayLengths)-1,len(guessRayLengths))
    f = plt.figure()
    plt.subplot(1,2,1)
    plt.title("r_abs_dot")
    plt.plot(t,r_abs_dot)
    plt.subplot(1,2,2)
    plt.title("theta_dot^2")
    plt.plot(t,theta_dot**2)
    f.savefig("guessderivs.png", dpi=300)

    zeroplaces = np.argwhere(np.abs(r_abs_dot)<0.03)
    print(zeroplaces.shape)
    f = plt.figure()
    plt.plot(q[:,1], -1*q[:,0])
    plt.scatter(q[zeroplaces,1], -1*q[zeroplaces,0], label="r_abs_dot~=0", color="r")
    plt.legend()
    f.savefig("zeroplaces.png", dpi=300)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef reconstruct_iterate_implicit(np.ndarray[np.double_t, ndim=2] q_mean, np.ndarray[np.double_t, ndim=1] guessRayLengths):
    cdef np.ndarray[np.double_t, ndim=1] r_abs = guessRayLengths.copy()
    cdef np.ndarray[np.double_t, ndim=2] u = q_mean/magnitude(q_mean).reshape(q_mean.shape[0],1)
    cdef np.ndarray[np.double_t, ndim=2] r = np.empty_like(q_mean)
    cdef np.ndarray[np.double_t, ndim=2] r_dot
    cdef np.ndarray[np.double_t, ndim=1] r_dot_abs
    cdef np.ndarray[np.double_t, ndim=2] e
    cdef np.ndarray[np.double_t, ndim=1] q_abs = magnitude(q_mean)
    cdef np.ndarray[np.double_t, ndim=1] diffs = np.empty_like(q_abs)
    cdef np.ndarray[np.double_t, ndim=1] ue_inner = np.empty_like(q_abs)
    cdef np.ndarray[np.double_t, ndim=1] q_abs_sq_ue = np.empty_like(q_abs)  # = q_abs**2 * ue_inner
    cdef np.ndarray[np.double_t, ndim=1] q_abs_sq_ue_cumsum  # cumulative sum of q_abs_sq_ue
    cdef double first, second, third, q_abs_sq_ue_sum
    cdef int i, t, j
    cdef double init_len = 0
    cdef double loop_len = 0
    cdef double r_abs_, u_, r_dot_abs_, e1_, e2_, q_abs_, diffs_
    for i in range(100):
        q_abs_sq_ue_sum = 0
        for j in prange(r_abs.shape[0], nogil=True):
            r[j, 0] = u[j, 0] * r_abs[j]
            r[j, 1] = u[j, 1] * r_abs[j]
        r_dot = dt(r, 1)
        for j in prange(r_abs.shape[0], nogil=True):
            r_dot_abs_ = sqrt(r_dot[j, 0]*r_dot[j, 0] + r_dot[j, 1]*r_dot[j, 1])
            e1_ = r_dot[j, 0]/r_dot_abs_
            e2_ = r_dot[j, 1]/r_dot_abs_
            ue_inner[j] = fabs(e1_*u[j, 0]+e2_*u[j, 1])
            q_abs_sq_ue[j] = q_abs[j]*q_abs[j]*ue_inner[j]
            q_abs_sq_ue_sum += q_abs_sq_ue[j]
        q_abs_sq_ue_cumsum = np.cumsum(q_abs_sq_ue)
        for t in prange(q_abs.shape[0], nogil=True):
        # for t in range(q_abs.shape[0]):
            first = 3.0*q_abs_sq_ue_cumsum[t]
            # print("msg1")
            second = (( q_abs_sq_ue[0] / fabs(u[0, 0]*r_dot[0, 0]+u[0, 1]*r_dot[0, 1]))**(3/2))
            # print("msg2")
            third = -3.0*q_abs_sq_ue_sum
            # print("msg3")
            diffs[t] = (fabs(cbrt(first+second+third)) -r_abs[t])
            # print("msg4")
        maxdiff = np.max(diffs)
        if maxdiff>1:
            diffnorm = diffs/maxdiff
        else:
            diffnorm = diffs
        r_abs += diffnorm*0.01

    return r_abs, []


def rk_f(t, y, Q,Q_dot,theta,theta_dot):
    #print("t = "+str(t))
    # (np.sign(theta_dot[t])*dt(theta,2)[t])/np.abs(theta_dot[t])
    theta_ddot = dt(theta,2)

    return 3*np.abs(theta_dot[t])+np.tan(y)*( theta_ddot[t]/theta_dot[t] - (2*Q_dot[t]/Q[t]))

def reconstruct_rk(q_mean, guessRayLengths):
    dirs = q_mean/magnitude(q_mean).reshape(q_mean.shape[0],1)
    Q = magnitude(q_mean)
    Q_dot = dt(Q,1)
    theta = np.arctan2(q_mean[:,1],q_mean[:,0])
    theta_dot = dt(theta,1)
    theta_ddot = dt(theta,2)
    r_guess = dirs*guessRayLengths.reshape(guessRayLengths.shape[0],1)
    r_guess_dot = dt(r_guess,1)
    velo_guess = magnitude(r_guess_dot)


    t = np.linspace(0,Q.shape[0], num=Q.shape[0])
    eqpart = np.sign(theta_dot)*theta_ddot/np.abs(theta_dot)

    '''plt.close('all')
    plt.subplot(1,3,1)
    plt.plot(t,theta, label="theta")
    plt.subplot(1,3,2)
    plt.plot(t,theta_dot, label="theta_dot")
    plt.subplot(1,3,3)
    plt.plot(t,theta_ddot, label="theta_ddot")
    plt.legend()
    plt.show()'''

    v_guess = np.arccos(np.sign(theta_dot) * innerProduct(r_guess,r_guess_dot)/(guessRayLengths*velo_guess))

    '''plt.close('all')
    plt.plot(t,v_guess, label="v guess")
    plt.legend()
    plt.show()'''

    t0 = 100
    v0 = v_guess[t0]
    print("v0 = "+str(v0))
    t, my_guess_v, tans = rk4(rk_f,t0,v0,Q,Q_dot,theta,theta_dot,2,399)

    

    #------------- ver 2
    #my_guess_y = my_guess_v
    #sinv = np.sqrt(1/(1+1/(my_guess_y)**2))

    #--------------------


    r_lengths = np.cbrt(np.sin(my_guess_v)*((Q[t0:len(my_guess_v)+t0]**2)/np.abs(theta_dot[t0:len(my_guess_v)+t0])))
    #print(my_guess_v)
    curve_guess = dirs[t0:2*len(my_guess_v)+t0:2]*r_lengths.reshape(r_lengths.shape[0],1)

    f = plt.figure()
    plt.subplot(1,2,1)
    plt.plot(curve_guess[:,1],-1*curve_guess[:,0], label="rk")
    plt.plot(r_guess[t0:len(my_guess_v)+t0,1],-1*r_guess[t0:len(my_guess_v)+t0,0], label="guess")
    plt.legend()

    t = np.linspace(0,len(my_guess_v),num=len(my_guess_v))
    plt.subplot(1,2,2)
    plt.plot(t,my_guess_v)
    f.savefig("rk.png", dpi=300)
    plt.close(f)
    
    


def makePeriodic(arr):
    if len(arr.shape)>1:
        ret_arr = np.ndarray((arr.shape[0]+1,2))
    else:
        ret_arr = np.ndarray((arr.shape[0]+1,))
    
    ret_arr[:arr.shape[0]] = arr
    ret_arr[-1] = arr[0]

    return ret_arr

def dydt(t,y):
    return 3*np.abs(thetadot_interp.interpolator(t))+np.tan(y)*( thetaddot_interp.interpolator(t)/thetadot_interp.interpolator(t)- (2*Qdot_interp.interpolator(t)/Q_interp.interpolator(t)) )

def bc(ya,yb):
    return ya-yb

def reconstruct_bvp(q_mean, guessRayLengths):
    dirs = q_mean/magnitude(q_mean).reshape(q_mean.shape[0],1)
    Q = magnitude(q_mean)
    Q_dot = dt(Q,1)
    theta = np.arctan2(q_mean[:,1],q_mean[:,0])
    theta_dot = dt(theta,1)
    theta_ddot = dt(theta,2)
    r_guess = dirs*guessRayLengths.reshape(guessRayLengths.shape[0],1)
    r_guess_dot = dt(r_guess,1)
    velo_guess = magnitude(r_guess_dot)
    v_guess = np.arccos(np.sign(theta_dot) * innerProduct(r_guess,r_guess_dot)/(guessRayLengths*velo_guess))

    Q = makePeriodic(Q)
    Q_dot = makePeriodic(Q_dot)
    theta = makePeriodic(theta)
    theta_dot = makePeriodic(theta_dot)
    theta_ddot = makePeriodic(theta_ddot)
    r_guess = makePeriodic(r_guess)
    r_guess_dot = makePeriodic(r_guess_dot)
    velo_guess = makePeriodic(velo_guess)
    v_guess = makePeriodic(v_guess)
    v_guess = v_guess.reshape(1,v_guess.shape[0])

    t = np.linspace(0,Q.shape[0], num=Q.shape[0])
    Q_interp.setInterpolator(t,Q)
    Qdot_interp.setInterpolator(t,Q_dot)
    theta_interp.setInterpolator(t,theta)
    thetadot_interp.setInterpolator(t,theta_dot)
    thetaddot_interp.setInterpolator(t,theta_ddot)

    t0 = 100

    res = solve_bvp(dydt,bc,t,v_guess, max_nodes=100000)
    print("status of result: "+str(res.status))

    tres = res.x
    vres = res.y.reshape(res.y.shape[1],)
    R = np.cbrt( ((Q_interp.interpolator(tres)**2)/np.abs(Qdot_interp.interpolator(tres))) * np.sin(vres))
    f = plt.figure()
    plt.plot(tres,R)
    f.savefig("rtest.png", dpi=300)
    plt.close(f)
    f = plt.figure()
    plt.plot(tres,vres)
    f.savefig("vtest.png", dpi=300)
    plt.close(f)


