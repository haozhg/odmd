import numpy as np


class WindowDMD:
    """WindowDMD is a class that implements window dynamic mode decomposition
    The time complexity (for one iteration) is O(n^2), and space complexity is 
    O(n^2), where n is the state dimension

    Algorithm description:
        At time step k, define two matrix 
        X(k) = [x(k-w+1),x(k-w+2),...,x(k)], Y(k) = [y(k-w+1),y(k-w+2),...,y(k)], 
        that contain the recent w snapshot pairs from a finite time window, 
        where x(k), y(k) are the n dimensional state vector, 
        y(k) = f(x(k)) is the image of x(k), f() is the dynamics. 
        Here, if the (discrete-time) dynamics are given by z(k) = f(z(k-1)), then x(k), y(k)
        should be measurements correponding to consecutive states z(k-1) and z(k).
        At time k+1, we need to forget the old snapshot pair xold = x(k-w+1), yold = y(k-w+1), 
        and remember the new snapshot pair xnew = x(k+1), ynew = y(k+1)
        We would like to update the DMD matrix Ak = Yk*pinv(Xk) recursively 
        by efficient rank-2 updating window DMD algrithm.
        
    Usage:  
        wdmd = WindowDMD(n,windowsize)
        wdmd.initialize(Xq,Yq)
        wdmd.update(xold,yold,xnew,ynew)
        evals, modes = wdmd.computemodes()
            
    properties:
        n: state dimension
        windowsize: window size
        timestep: number of snapshot pairs processed (i.e., the current time step)
        A: DMD matrix, size n by n
        P: Matrix that contains information about recent w snapshots, size n by n

    methods:
        initialize(Xq, Yq), initialize window DMD algorithm
        update(x,y), update DMD computation when new snapshot pair (x,y) becomes available
        computemodes(), compute and return DMD eigenvalues and DMD modes
        
    Authors:
        Hao Zhang
        Clarence W. Rowley
        
    References:
        Hao Zhang, Clarence W. Rowley, Eric A. Deem, and Louis N. Cattafesta,
        ``Online Dynamic Mode Decomposition for Time-varying Systems",  
        in production, 2017. Available on arXiv.
    
    Date created: April 2017
    
    To import the WindowDMD class, add import window at head of Python scripts.
    To look up this documentation, type help(window.WindowDMD) or window.WindowDMD?
    """
    def __init__(self, n=0, windowsize=0, timestep=0, A=None, P=None):
        """
        Creat an object for window DMD
        Usage: wdmd = WindowDMD(n,windowsize)
            """
        self.n = n
        self.windowsize = windowsize
        self.timestep = timestep
        if A is None or P is None:
            self.A = np.zeros([n,n])
            self.P = np.zeros([n,n])
        else:
            self.A = A
            self.P = P

    def initialize(self, Xq, Yq):
        """Initialize window DMD with first q snapshot pairs stored in (Xq, Yq)
        Usage: wdmd.initialize(Xq,Yq)
        """
        q = len(Xq[0,:])
        if self.timestep == 0 and self.windowsize == q and self.windowsize >= self.n + 1:
            self.A = Yq.dot(np.linalg.pinv(Xq))
            self.P = np.linalg.inv(Xq.dot(Xq.T))
            self.timestep += q
        
    def update(self, xold, yold, xnew, ynew):
        """Update the DMD computation by sliding the finite time window forward
        Forget the oldest pair of snapshots (xold, yold), and remembers the newest 
        pair of snapshots (xnew, ynew) in the new time window. If the new finite 
        time window at time step k+1 includes recent w snapshot pairs as
        X(k+1) = [x(k-w+2),x(k-w+3),...,x(k+1)], Y(k+1) = [y(k-w+2),y(k-w+3),...,y(k+1)], 
        where y(k) = f(x(k)) and f is the dynamics, then we should take
        xold = x(k-w+1), yold = y(k-w+1), xnew = x(k+1), ynew = y(k+1)
        Usage: wdmd.update(xold, yold, xnew, ynew)
        """
        # direct rank-2 update
        # define matrices
        U, V = np.column_stack((xold, xnew)), np.column_stack((yold, ynew))
        C = np.diag([-1,1])
        # compute PkU matrix vector product beforehand
        PkU = self.P.dot(U)
        # compute AkU matrix vector product beforehand
        AkU = self.A.dot(U)
        # compute Gamma
        Gamma = np.linalg.inv(C+U.T.dot(PkU))
        # update A
        self.A += (V-AkU).dot(Gamma).dot(PkU.T)
        # update P
        self.P += -PkU.dot(Gamma).dot(PkU.T)
        # ensure P is SPD by taking its symmetric part
        self.P = (self.P + self.P.T)/2
        
        # time step + 1
        self.timestep += 1

    def computemodes(self):
        """Compute and return DMD eigenvalues and DMD modes at current time step
        Usage: evals, modes = wdmd.computemodes()
        """
        evals, modes = np.linalg.eig(self.A)
        return evals, modes