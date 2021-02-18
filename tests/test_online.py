import numpy as np
from odmd import OnlineDMD

EPS = 1e-6


def test_online():
    for n in range(2, 10):
        m = 100 * n
        A = np.random.randn(n, n)
        X = np.random.randn(n, m)
        Y = A.dot(X)

        # run online DMD
        # no need to initialize
        onlinedmd = OnlineDMD(n)
        for i in range(m):
            onlinedmd.update(X[:, i], Y[:, i])
            if i >= 2 * n:
                assert np.linalg.norm(onlinedmd.A - A) < EPS
