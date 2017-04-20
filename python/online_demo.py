"""
An example to demonstrate online dynamic mode decomposition

We take a 2D time varying system given by dx/dt = A(t)x
where x = [x1,x2]', A(t) = [0,w(t);-w(t),0], 
w(t)=1+epsilon*t, epsilon=0.1. The slowly time varying eigenvlaues of A(t)
are pure imaginary, i.e, +(1+0.1t)j and -(1+0.1t)j, where j is the imaginary unit

At time step k, define two matrix Xk = [x(1),x(2),...,x(k)], Yk = [y(1),y(2),...,y(k)],
that contain all the past snapshot pairs, we would like to compute 
Ak = Yk*pinv(Xk). This can be done by brute-force batch DMD, 
and by efficient rank-1 updating online DMD algrithm.

Batch DMD computes DMD matrix by brute-force taking the pseudo-inverse directly

Online DMD computes the DMD matrix by using efficient rank-1 update idea

We compare the performance of online DMD (with lambda=1,0.9) with the brute-force batch DMD
approach in terms of tracking time varying eigenvalues, by comparison with the analytical solution

Authors: Hao Zhang, Princeton University
         haozhang@princeton.edu
    
Date created: April 2017
"""


import sys
sys.path.append('..')

from online import OnlineDMD
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


# batch DMD
q = 20
AbatchDMD = np.empty((n,n,m))
evalsbatchDMD = np.empty((n,m),dtype=complex)
start = time.clock()
for k in range(q,m):
    AbatchDMD[:,:,k] = y[:,:k].dot(np.linalg.pinv(x[:,:k]))
    evalsbatchDMD[:,k] = np.log(np.linalg.eigvals(AbatchDMD[:,:,k]))/dt
end = time.clock()
print "Batch DMD, time = " + str(end-start) + " secs"


# Online DMD, forgetting = 1
q = 20
evalsonlineDMD1 = np.empty((n,m),dtype=complex)
odmd = OnlineDMD(n,1.0)
odmd.initialize(x[:,:q],y[:,:q])
start = time.clock()
for k in range(q,m):
    odmd.update(x[:,k],y[:,k])
    evalsonlineDMD1[:,k] = np.log(np.linalg.eigvals(odmd.A))/dt
end = time.clock()
print "Online DMD, forgetting = 1, time = " + str(end-start) + " secs"


# Online DMD, forgetting = 0.9
q = 20
evalsonlineDMD09 = np.empty((n,m),dtype=complex)
odmd = OnlineDMD(n,0.9)
odmd.initialize(x[:,:q],y[:,:q])
start = time.clock()
for k in range(q,m):
    odmd.update(x[:,k],y[:,k])
    evalsonlineDMD09[:,k] = np.log(np.linalg.eigvals(odmd.A))/dt
end = time.clock()
print "Online DMD, forgetting = 0.9, time = " + str(end-start) + " secs"


# visualize true, batch, online (forgettting=1,0.9)
plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(t, np.imag(evals[0,:]), 'k-',label='true',linewidth=2.0)
plt.plot(t[q:], np.imag(evalsbatchDMD[0,q:]), 'r-',label='batch',linewidth=2.0)
plt.plot(t[q:], np.imag(evalsonlineDMD1[0,q:]), 'g--',label='online, $\lambda$=1',linewidth=2.0)
plt.plot(t[q:], np.imag(evalsonlineDMD09[0,q:]), 'b-',label='online, $\lambda$=0.9',linewidth=2.0)
plt.tick_params(labelsize=20)
plt.xlabel('Time', fontsize=20)
plt.ylabel('Im', fontsize=20)
plt.title('Imignary part of eigenvalues', fontsize=20)
plt.legend(loc='best', fontsize=20, shadow=True)
plt.xlim([0,10])
plt.ylim([1,2])
plt.show()