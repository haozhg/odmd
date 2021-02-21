import numpy as np
from odmd import OnlineDMD

EPS = 1e-6


def test_online():
    # n is state dimension
    for n in range(2, 10):
        T = 100 * n  # number of measurements
        A = np.random.randn(n, n)
        X = np.random.randn(n, T)
        Y = A.dot(X)

        # run online DMD
        # no need to initialize
        onlinedmd = OnlineDMD(n, 0.9)
        for t in range(T):
            onlinedmd.update(X[:, t], Y[:, t])
            if onlinedmd.ready:
                assert np.linalg.norm(onlinedmd.A - A) < EPS
