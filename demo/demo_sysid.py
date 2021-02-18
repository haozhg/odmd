# -*- coding: utf-8 -*-
"""
An example to demonstrate online linear system identification
We demonstrate the use of OnlineSysId class with a simple linear system.
Take a 2D time-varying system dz/dt=A(t)z(t)+B(t)u(t), where A(t) and B(t)
are slowly varying with time. In particular, we take A(t)=(1+eps*t)*A0,
B(t)=(1+eps*t)*B0, and eps = 0.1 is small. It is discretize with
time step dt = 0.1. Denote the discrete system as z(k)=A(k)z(k-1)+
B(k)u(k-1). We define x(k) = [z(k-1);u(k-1)], y(k) = z(k),
F(k)=[A(k),B(k)], then the original system can be written as y(k)=F(k)x(k).
At time step k, define two matrix X(k) = [x(1),x(2),...,x(k)], 
Y(k) = [y(1),y(2),...,y(k)], that contain all the past snapshot pairs.
The best fit to the data is Fk = Yk*pinv(Xk).
An exponential weighting factor rho=sigma^2 (0<rho<=1) that places more 
weight on recent data can be incorporated into the definition of X(k) and
Y(k) such that X(k) = [sigma^(k-1)*x(1),sigma^(k-2)*x(2),...,
sigma^(1)*x(k-1),x(k)], Y(k) = [sigma^(k-1)*y(1),sigma^(k-2)*y(2),...,
sigma^(1)*y(k-1),y(k)].
At time step k+1, we need to include new snapshot pair x(k+1), y(k+1).
We would like to update the DMD matrix Fk = Yk*pinv(Xk) recursively 
by efficient rank-1 updating online DMD algorithm.
Authors: 
Hao Zhang
Clarence W. Rowley
Reference:
Hao Zhang, Clarence W. Rowley,
``Real-time control of nonlinear and time-varying systems based on 
online linear system identification", in production, 2017.
Created:
June 2017.
"""


import matplotlib.pyplot as plt
import numpy as np
from onlinesysid import OnlineSysId
import sys
sys.path.append('..')


# define dynamics
A0 = np.array([[0, 1], [-1, -0.1]])
B0 = np.array([[0], [1]])
epsilon = 1e-1


def dyn(t, z, u):
    dzdt = (1+epsilon*t)*A0.dot(z) + (1+epsilon*t)*B0.dot(u)
    return dzdt


# set up simulation parameter
dt = 0.1
tmax, tc = 10, 0.5
kmax, kc = int(tmax/dt), int(tc/dt)
tspan = np.linspace(0, tmax, kmax+1)
# control input sine wave
gamma = 1
omega = 5
# dimensions
n = 2
p = 1
q = n + p
# online linear system identification setup
weighting = 0.01**(2.0/kc)
osysid = OnlineSysId(n, q, weighting)
osysid.initializeghost()
# store data mtrices
x, y = np.zeros([q, kmax]), np.zeros([n, kmax])
Aerror, Berror = np.zeros(kmax), np.zeros(kmax)

# initial condition,state and control
z0 = np.array([[1], [0]])
u0 = np.array([[0.0]])
zk, uk = z0, u0
# system simulation
for k in range(kmax):
    # update state x(k) = [z(k-1);u(k-1)]
    x[:, k] = np.vstack((zk, uk)).reshape(q)
    # forward the system for one step
    zk = zk + dt*dyn(k*dt, zk, uk)
    # update control input according to sine wave
    uk = np.array([[gamma*np.sin(omega*(k+1)*dt)]])
    # update state y(k) = z(k)
    y[:, k] = zk.reshape(n)
    # use new data to update online system identification
    osysid.update(x[:, k], y[:, k])
    # model error at time k
    Ak = np.identity(n)+dt*(1+epsilon*(k+1)*dt)*A0
    Bk = dt*(1+epsilon*(k+1)*dt)*B0
    Aerror[k] = np.linalg.norm(
        osysid.A[:, :n]-Ak, 'fro')/np.linalg.norm(Ak, 'fro')
    Berror[k] = np.linalg.norm(
        osysid.A[:, n:]-Bk, 'fro')/np.linalg.norm(Bk, 'fro')

# visualize snapshots
plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(tspan[1:], y[0, :], 'bs-', linewidth=2.0,  label='$z_1(t)$')
plt.plot(tspan[1:], y[1, :], 'g^-', linewidth=2.0,  label='$z_2(t)$')
plt.plot(tspan[1:], x[2, :], 'rd-', linewidth=2.0,  label='$u(t)$')
plt.legend(loc='best', fontsize=20, shadow=True)
plt.xlabel('Time', fontsize=20)
plt.title('Snapshots', fontsize=20)
plt.tick_params(labelsize=20)
plt.grid()
plt.xlim([0, 10])
plt.ylim([-2, 2])
plt.show()

# visualize model error
plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(tspan[1:], Aerror, 'bs-', linewidth=2.0,  label='\% error in $A(t)$')
plt.plot(tspan[1:], Berror, 'g^-', linewidth=2.0,  label='\% error in $B(t)$')
plt.legend(loc='best', fontsize=20, shadow=True)
plt.xlabel('Time', fontsize=20)
plt.title('Online DMD model error', fontsize=20)
plt.tick_params(labelsize=20)
plt.grid()
plt.xlim([0, 10])
plt.ylim([0, 0.2])
plt.show()
