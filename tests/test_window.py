import numpy as np
from odmd import WindowDMD

EPS = 1e-6


def test_window():
    for n in range(2, 10):
        T = 100 * n
        A = np.random.randn(n, n)
        X = np.random.randn(n, T)
        Y = A.dot(X)

        # run window DMD
        windowdmd = WindowDMD(n, 2 * n)
        # initialize
        windowdmd.initialize(X[:, : 2 * n], Y[:, : 2 * n])
        # online update
        for t in range(2 * n, T):
            windowdmd.update(X[:, t], Y[:, t])
            if t >= 2 * n:
                assert np.linalg.norm(windowdmd.A - A) < EPS
