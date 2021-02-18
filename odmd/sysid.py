# -*- coding: utf-8 -*-
import numpy as np


class OnlineSysId:
    """OnlineSysID is a class that implements the online system identification.
    The time complexity (multiply-add operation for one iteration) is O(4n^2)
    , and space complexity is O(2n^2), where n is the problem dimension.
    It works for both online linear and nonlinear system identification.

    Algorithm description:
    Suppose that we have a nonlinear and time-varying system z(k) =
    f(t(k-1),z(k-1),u(k-1)). We aim to build local linear model in the form
    of z(k) = F(z(k),t(k)) z(k-1) + G(z(k),t(k)) u(k-1), where F(z(k),t(k)), 
    G(z(k),t(k)) are two matrices. We define x(k) = [z(k-1); u(k-1)], y(k) =
    z(k), and A(k) = [F(z(k),t(k)), G(z(k),t(k))]. Then the local linear 
    model can be written as y(k) = A(k) x(k).
    We can also build nonlinear model by defining nonlinear observable x(k)
    of state z(k-1) and control u(k-1). For example, for a nonlinear system 
    z(k)=z(k-1)^2+u(k-1)^2, we can define x(k) = [z(k-1)^2;u(k-1)^2], y(k) =
    z(k), then it can be written in linear form y(k) = A*x(k), where A=[1,1].
    At time step k, we assume that we have access to z(j),u(j),j=0,1,2,...k.
    Then we have access to x(j), y(j), j=1,1,2,...k. We define two matrices
    X(k) = [x(1),x(2),...,x(k)], Y(k) = [y(1),y(2),...,y(k)], that contain 
    all the past snapshot. The best fit to the data is Ak = Yk*pinv(Xk).
    An exponential weighting factor rho=sigma^2 (0<rho<=1) that places more 
    weight on recent data can be incorporated into the definition of X(k) and
    Y(k) such that X(k) = [sigma^(k-1)*x(1),sigma^(k-2)*x(2),...,
    sigma^(1)*x(k-1),x(k)], Y(k) = [sigma^(k-1)*y(1),sigma^(k-2)*y(2),...,
    sigma^(1)*y(k-1),y(k)].
    At time step k+1, we need to include new snapshot pair x(k+1), y(k+1).
    We would like to update the general DMD matrix Ak = Yk*pinv(Xk) recursively 
    by efficient rank-1 updating online DMD algorithm.
    Therefore, A(k) explains the most recent data and is considered to be 
    the local linear model for the original nonlinear and/or time-varying 
    system. This local linear model can be used for short-horizon prediction 
    and real-time control.

    Usage:
    osysid = OnlineSysId(n,q,weighting)
    osysid.initialize(X0,Y0)
    osysid.initilizeghost()
    osysid.update(x,y)

    properties:
    n: state dimension
    q: observable vector dimension, state dimension + control dimension for
    linear system identification case
    weighting: weighting factor in (0,1]
    timestep: number of snapshot pairs processed
    A: general DMD matrix, size n by q
    P: matrix that contains information about past snapshots, size q by q

    methods:
    initialize(X0, Y0), initialize online system identification with k0 snapshot
                        pairs stored in (X0, Y0)
    initializeghost(),  initialize online system identification with epsilon 
                        small (1e-15) ghost snapshot pairs before t=0
    update(x,y), update when new snapshot pair (x,y) becomes available

    Authors: 
    Hao Zhang
    Clarence W. Rowley

    Reference:
    Hao Zhang, Clarence W. Rowley,
    ``Real-time control of nonlinear and time-varying systems based on 
    online linear system identification", in production, 2017.

    Created:
    June 2017.

    To import the OnlineSysId class, add import OnlineSysId at head of Python scripts.
    To look up this documentation, type help(onlinesysid.OnlineSysId) or 
    onlinesysid.OnlineSysId?
    """

    def __init__(self, n=0, q=0, weighting=1):
        """
        Creat an object for online system identification class OnlineSysId
        Usage: osysid = OnlineSysId(n,q,weighting)
            """
        self.n = n
        self.q = q
        self.weighting = weighting
        self.timestep = 0
        self.A = np.zeros([n, q])
        self.P = np.zeros([q, q])

    def initialize(self, X0, Y0):
        """Initialize OnlineSysId with first k0 snapshot pairs stored in (X0, Y0)
        Usage: osysid.initialize(X0,Y0)
        """
        k0 = len(X0[0, :])
        if self.timestep == 0 and np.linalg.matrix_rank(X0) == self.q:
            weight = np.sqrt(self.weighting)**range(k0-1, -1, -1)
            X0hat, Y0hat = weight*X0, weight*Y0
            self.A = Y0hat.dot(np.linalg.pinv(X0hat))
            self.P = np.linalg.inv(X0hat.dot(X0hat.T))/self.weighting
            self.timestep += k0

    def initializeghost(self):
        """Initialize online DMD with epsilon small (1e-15) ghost snapshot pairs 
        before t=0
        Usage: osysid.initilizeghost()
        """
        epsilon = 1e-15
        self.A = np.zeros([self.n, self.q])
        self.P = (1.0/epsilon)*np.identity(self.q)

    def update(self, x, y):
        """Update online DMD computation with a new pair of snapshots (x,y)
        Here, if the (discrete-time) dynamics are given by z(k) = 
        f(t(k-1),z(k-1),u(k-1)), then x=[z(k-1);u(k-1)], y=z(k).
        Usage: osysid.update(x, y)
        """
        # compute P*x matrix vector product beforehand
        Px = self.P.dot(x)
        # compute gamma
        gamma = 1.0/(1 + x.T.dot(Px))
        # update A
        self.A += np.outer(gamma*(y-self.A.dot(x)), Px)
        # update P, group Px*Px' to ensure positive definite
        self.P = (self.P - gamma*np.outer(Px, Px))/self.weighting
        # ensure P is SPD by taking its symmetric part
        self.P = (self.P + self.P.T)/2
        # time step + 1
        self.timestep += 1
