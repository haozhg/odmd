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
        w = 2 * n
        windowdmd = WindowDMD(n, w, 0.9)
        # initialize
        windowdmd.initialize(X[:, : w], Y[:, : w])
        # online update
        for t in range(2 * n, T):
            windowdmd.update(X[:, t], Y[:, t])
            if windowdmd.ready:
                assert np.linalg.norm(windowdmd.A - A) < EPS
