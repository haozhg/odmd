import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class WindowDMD:
    """WindowDMD is a class that implements window dynamic mode decomposition
    The time complexity (multiply–add operation for one iteration) is O(8n^2), 
    and space complexity is O(2wn+2n^2), where n is the state dimension, w is 
    the window size.

    Algorithm description:
        At time step k, define two matrix X(k) = [x(k-w+1),x(k-w+2),...,x(k)], 
        Y(k) = [y(k-w+1),y(k-w+2),...,y(k)], that contain the recent w snapshot 
        pairs from a finite time window, where x(k), y(k) are the n dimensional 
        state vector, y(k) = f(x(k)) is the image of x(k), f() is the dynamics. 
        Here, if the (discrete-time) dynamics are given by z(k) = f(z(k-1)), 
        then x(k), y(k) should be measurements correponding to consecutive 
        states z(k-1) and z(k).
        At time k+1, we need to forget the old snapshot pair xold = x(k-w+1), 
        yold = y(k-w+1), and remember the new snapshot pair x = x(k+1), 
        y = y(k+1).
        We would like to update the DMD matrix Ak = Yk*pinv(Xk) recursively 
        by efficient rank-2 updating window DMD algrithm.
        An exponential weighting factor can be used to place more weight on
        recent data.

    Usage:  
        wdmd = WindowDMD(n,w)
        wdmd.initialize(Xw,Yw) # this is necessary
        wdmd.update(x,y)
        evals, modes = wdmd.computemodes()

    properties:
        n: state dimension
        w: window size
        weighting: weighting factor in (0,1]
        timestep: number of snapshot pairs processed (i.e., current time step)
        Xw: recent w snapshots x stored in Xw, size n by w
        Yw: recent w snapshots y stored in Yw, size n by w
        A: DMD matrix, size n by n
        P: Matrix that contains information about recent w snapshots, size n by n

    methods:
        initialize(Xw, Yw), initialize window DMD algorithm with w snapshot pairs, this is necessary
        update(x,y), update DMD computation by adding a new snapshot pair
        computemodes(), compute and return DMD eigenvalues and DMD modes

    Authors:
        Hao Zhang
        Clarence W. Rowley

    References:
        Zhang, Hao, Clarence W. Rowley, Eric A. Deem, and Louis N. Cattafesta. 
        "Online dynamic mode decomposition for time-varying systems." 
        SIAM Journal on Applied Dynamical Systems 18, no. 3 (2019): 1586-1609.

    Date created: April 2017
    """

    def __init__(self, n: int, w: int, weighting: float = 1):
        """
        Creat an object for window DMD
        Usage: wdmd = WindowDMD(n,w,weighting)
            """
        assert n >= 1 and isinstance(n, int)
        assert w >= 1 and isinstance(w, int)
        assert weighting > 0 and weighting <= 1

        self.n = n
        self.w = w
        self.weighting = weighting
        self.timestep = 0
        self.Xw = deque()
        self.Yw = deque()
        self.A = np.zeros([n, n])
        self.P = np.zeros([n, n])
        # need to call initialize before update() and computemodes()
        self.ready = False

    def initialize(self, Xw, Yw):
        """Initialize window DMD with first w snapshot pairs stored in (Xw, Yw)
        Usage: wdmd.initialize(Xw,Yw)
        """
        assert Xw is not None and Yw is not None
        Xw, Yw = np.array(Xw), np.array(Yw)
        assert Xw.shape == Yw.shape
        assert Xw.shape == (self.n, self.w)

        # initialize Xw, Yw queue
        for i in range(self.w):
            self.Xw.append(Xw[:, i])
            self.Yw.append(Yw[:, i])

        # initialize A, P
        q = len(Xw[0, :])
        if self.timestep == 0 and self.w == q \
                and np.linalg.matrix_rank(Xw) == self.n:
            weight = np.sqrt(self.weighting)**range(q-1, -1, -1)
            Xwhat, Ywhat = weight*Xw, weight*Yw
            self.A = Ywhat.dot(np.linalg.pinv(Xwhat))
            self.P = np.linalg.inv(Xwhat.dot(Xwhat.T))/self.weighting
            self.timestep += q
        self.ready = True

    def update(self, x, y):
        """Update the DMD computation by sliding the finite time window forward
        Forget the oldest pair of snapshots (xold, yold), and remembers the newest 
        pair of snapshots (x, y) in the new time window. If the new finite 
        time window at time step k+1 includes recent w snapshot pairs as
        X(k+1) = [x(k-w+2),x(k-w+3),...,x(k+1)], Y(k+1) = [y(k-w+2),y(k-w+3),
        ...,y(k+1)], where y(k) = f(x(k)) and f is the dynamics, then we should 
        take x = x(k+1), y = y(k+1)
        Usage: wdmd.update(x, y)
        """
        if not self.ready:
            raise Exception(
                "Not initialized yet! Need to call self.initialize(Xw, Yw)")

        assert x is not None and y is not None
        x, y = np.array(x), np.array(y)

        assert np.array(x).shape == np.array(y).shape
        assert np.array(x).shape[0] == self.n

        # define old snapshots to be discarded
        xold, yold = self.Xw.popleft(), self.Yw.popleft()
        # Update recent w snapshots
        self.Xw.append(x)
        self.Yw.append(y)

        # direct rank-2 update
        # define matrices
        U, V = np.column_stack((xold, x)), np.column_stack((yold, y))
        C = np.diag([-(self.weighting)**(self.w), 1])
        # compute PkU matrix matrix product beforehand
        PkU = self.P.dot(U)
        # compute AkU matrix matrix product beforehand
        AkU = self.A.dot(U)
        # compute Gamma
        Gamma = np.linalg.inv(np.linalg.inv(C)+U.T.dot(PkU))
        # update A
        self.A += (V-AkU).dot(Gamma).dot(PkU.T)
        # update P
        self.P = (self.P - PkU.dot(Gamma).dot(PkU.T))/self.weighting
        # ensure P is SPD by taking its symmetric part
        self.P = (self.P + self.P.T)/2

        # time step + 1
        self.timestep += 1

    def computemodes(self):
        """Compute and return DMD eigenvalues and DMD modes at current time step
        Usage: evals, modes = wdmd.computemodes()
        """
        if not self.ready:
            raise Exception(
                "Not initialized yet! Need to call self.initialize(Xw, Yw)")
        evals, modes = np.linalg.eig(self.A)
        return evals, modes
