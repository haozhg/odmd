import numpy as np


class OnlineDMD:
    """OnlineDMD is a class that implements online dynamic mode decomposition
    The time complexity (for one iteration) is O(n^2), and space complexity is 
    O(n^2), where n is the state dimension.
    
    Algorithm description:
        At time step k, define two matrix Xk = [x(1),x(2),...,x(k)], Yk = [y(1),y(2),...,y(k)],
        that contain all the past snapshot pairs, where x(k), y(k) are the n 
        dimensional state vector, y(k) = f(x(k)) is the image of x(k), f() is the dynamics. 
        Here, if the (discrete-time) dynamics are given by z(k) = f(z(k-1)), then x(k), y(k)
        should be measurements correponding to consecutive states z(k-1) and z(k).
        We would like to update the DMD matrix Ak = Yk*pinv(Xk) recursively 
        by efficient rank-1 updating online DMD algrithm.
    
    Usage:
        odmd = OnlineDMD(n,forgetting)
        odmd.initialize(Xq,Yq)
        odmd.initilizeghost()
        odmd.update(x,y)
        evals, modes = odmd.computemodes()
            
    properties:
        n: state dimension
        forgetting: forgetting factor between (0,1]
        timestep: number of snapshot pairs processed (i.e., the current time step)
        A: DMD matrix, size n by n
        P: Matrix that contains information about past snapshots, size n by n

    methods:
        initialize(Xq, Yq), initialize online DMD algorithm with first q snapshot pairs stored in (Xq, Yq)
        initializeghost(), initialize online DMD algorithm with epsilon small (1e-15) ghost snapshot pairs before t=0
        update(x,y), update DMD computation when new snapshot pair (x,y) becomes available
        computemodes(), compute and return DMD eigenvalues and DMD modes
    
    Authors:
        Hao Zhang
        Clarence W. Rowley
        
    References:
        Hao Zhang, Clarence W. Rowley, Eric A. Deem, and Louis N. Cattafesta,
        ``Fast Quadratic-time Methods for Online Dynamic Mode Decomposition", 
        in production, 2017. To be submitted for publication, available on arXiv.
    
    Date created: April 2017
    
    To import the OnlineDMD class, add import online at head of Python scripts.
    To look up this documentation, type help(online.OnlineDMD) or online.OnlineDMD?
    """
    def __init__(self, n=0, forgetting=1, timestep=0, A=None, P=None):
        """
        Creat an object for online DMD
        Usage: odmd = OnlineDMD(n,forgetting)
            """
        self.n = n
        self.forgetting = forgetting
        self.timestep = timestep
        if A is None or P is None:
            self.A = np.zeros([n,n])
            self.P = np.zeros([n,n])
        else:
            self.A = A
            self.P = P

    def initialize(self, Xq, Yq):
        """Initialize online DMD with first q snapshot pairs stored in (Xq, Yq)
        Usage: odmd.initialize(Xq,Yq)
        """
        q = len(Xq[0,:])
        Xqhat, Yqhat = np.zeros(Xq.shape), np.zeros(Yq.shape)
        if self.timestep == 0 and self.n <= q:
            sqrtlambda = np.sqrt(self.forgetting)
            # multiply forgetting factor with snapshots
            for i in range(q):
                Xqhat[:,i] = Xq[:,i]*sqrtlambda**(q-1-i)
                Yqhat[:,i] = Yq[:,i]*sqrtlambda**(q-1-i)
            self.A = Yqhat.dot(np.linalg.pinv(Xqhat))
            self.P = np.linalg.inv(Xqhat.dot(Xqhat.T))/self.forgetting
            self.timestep += q
            
    def initializeghost(self):
        """Initialize online DMD with epsilon small (1e-15) ghost snapshot pairs before t=0
        Usage: odmd.initilizeghost()
        """
        epsilon=1e-15
        alpha = 1.0/epsilon
        self.A = np.random.randn(self.n, self.n)
        self.P = alpha*np.identity(self.n)
        
    def update(self, x, y):
        """Update the DMD computation with a new pair of snapshots (x,y)
        Here, if the (discrete-time) dynamics are given by z(k) = f(z(k-1)), then (x,y)
        should be measurements correponding to consecutive states z(k-1) and z(k).
        Usage: odmd.update(x, y)
        """
        # compute P*x matrix vector product beforehand
        Px = self.P.dot(x)
        # compute gamma
        gamma = 1.0/(1 + x.T.dot(Px))
        # update A
        self.A += gamma*np.outer(y-self.A.dot(x),Px)
        # update P
        self.P = (self.P - gamma*np.outer(Px,Px))/self.forgetting
        # time step + 1
        self.timestep += 1

    def computemodes(self):
        """Compute and return DMD eigenvalues and DMD modes at current time step
        Usage: evals, modes = odmd.computemodes()
        """
        evals, modes = np.linalg.eig(self.A)
        return evals, modes