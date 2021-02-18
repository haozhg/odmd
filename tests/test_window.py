import numpy as np
from odmd import WindowDMD


def test_window():
    for n in range(2, 10):
        m = 100 * n
        A = np.random.randn(n, n)
        X = np.random.randn(n, m)
        Y = A.dot(X)

        # run window DMD
        windowdmd = WindowDMD(n, 2 * n)
        # initialize
        windowdmd.initialize(X[:, :2 * n], Y[:, :2 * n])
        # online update
        for i in range(2 * n, m):
            windowdmd.update(X[:, i], Y[:, i])
            if i >= 2 * n:
                assert np.linalg.norm(windowdmd.A - A) < 1e-6
