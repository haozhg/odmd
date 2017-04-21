"""
An example to demonstrate window dynamic mode decomposition

We take a 2D time varying system given by dx/dt = A(t)x
where x = [x1,x2]', A(t) = [0,w(t);-w(t),0], 
w(t)=1+epsilon*t, epsilon=0.1. The slowly time varying eigenvlaues of A(t)
are pure imaginary, i.e, +(1+0.1t)j and -(1+0.1t)j, where j is the imaginary unit

At time step k, define two matrix Xk = [x(k-w+1),x(k-w+2),...,x(k)], Yk = [y(k-w+1),y(k-w+2),...,y(k)],
that contain the recent w snapshot pairs from a finite time window, 
we would like to compute Ak = Yk*pinv(Xk). This can be done by brute-force mini-batch DMD, 
and by efficient rank-2 updating window DMD algrithm.

Mini-batch DMD computes DMD matrix by taking the pseudo-inverse directly

Window DMD computes the DMD matrix by using efficient rank-2 update idea

We compare the performance of window DMD with the brute-force mini-batch DMD
approach in terms of tracking time varying eigenvalues, by comparison with the analytical solution
    
Authors: 
    Hao Zhang
    Clarence W. Rowley
    
Date created: April 2017
"""


import sys
sys.path.append('..')

from window import WindowDMD
import numpy as np
from scipy.integrate import odeint
import time
import matplotlib.pyplot as plt


# define dynamics
epsilon = 1e-1
def dyn(x,t):
    x1, x2 = x
    dxdt = [(1+epsilon*t)*x2,-(1+epsilon*t)*x1]
    return dxdt
# integrate from initial condition [1,0]    
tspan = np.linspace(0,10,101)
dt = 0.1
x0 = [1,0]
xsol = odeint(dyn,x0,tspan).T
x, y = xsol[:,:-1], xsol[:,1:]
t = tspan[:-1]
# true dynamics, true eigenvalues
n, m = len(x[:,0]), len(x[0,:])
A = np.empty((n,n,m))
evals = np.empty((n,m),dtype=complex)
for k in range(m):
    A[:,:,k] = np.array([[0,(1+epsilon*t[k])],[-(1+epsilon*t[k]),0]])
    evals[:,k] = np.linalg.eigvals(A[:,:,k])


# visualize snapshots
plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(tspan, xsol[0,:], 'bs-', linewidth=2.0,  label='$x_1(t)$')
plt.plot(tspan, xsol[1,:], 'g^-', linewidth=2.0,  label='$x_2(t)$')
plt.legend(loc='best',fontsize=20 ,shadow=True)
plt.xlabel('Time', fontsize=20)
plt.title('Snapshots', fontsize=20)
plt.tick_params(labelsize=20)
plt.grid()
plt.show()


# mini-batch DMD, w = 20
w = 20
AminibatchDMD = np.empty((n,n,m))
evalsminibatchDMD = np.empty((n,m),dtype=complex)
start = time.clock()
for k in range(w,m):
    AminibatchDMD[:,:,k] = y[:,k-w:k].dot(np.linalg.pinv(x[:,k-w:k]))
    evalsminibatchDMD[:,k] = np.log(np.linalg.eigvals(AminibatchDMD[:,:,k]))/dt
end = time.clock()
print "Mini-batch DMD, time = " + str(end-start) + " secs"


# Window DMD, w = 20
w = 20
evalswindowDMD = np.empty((n,m),dtype=complex)
wdmd = WindowDMD(n,w)
wdmd.initialize(x[:,:w],y[:,:w])
start = time.clock()
for k in range(w,m):
    wdmd.update(x[:,k-w+1],y[:,k-w+1],x[:,k],y[:,k])
    evalswindowDMD[:,k] = np.log(np.linalg.eigvals(wdmd.A))/dt
end = time.clock()
print "Window DMD, forgetting = 1, time = " + str(end-start) + " secs"


# visualize true, batch, window (forgettting=1,0.9)
plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(t, np.imag(evals[0,:]), 'k-',label='true',linewidth=2.0)
plt.plot(t[w:], np.imag(evalsminibatchDMD[0,w:]), 'r-',label='mini-batch, w=20',linewidth=2.0)
plt.plot(t[w:], np.imag(evalswindowDMD[0,w:]), 'g--',label='window, w=20',linewidth=2.0)
plt.tick_params(labelsize=20)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Im', fontsize=20)
plt.title('Imignary part of eigenvalues', fontsize=20)
plt.legend(loc='best', fontsize=20, shadow=True)
plt.xlim([0,10])
plt.ylim([1,2])
plt.show()