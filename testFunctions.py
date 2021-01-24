from scipy.optimize import rosen
import numpy as np


def testFcn1(x):
    """
    testFcn1
    """

    s1 = np.sum([(2 * (i + 1) - 1) * np.power(x[i] - (2 + (i + 1)), 4)
                 for i in range(len(x))], axis=0)
    s2 = np.sum([np.power(x[i] - x[j] + (j - i), 4) for i in range(len(x) - 1)
                 for j in range(i + 1, len(x))], axis=0)

    return s1 + s2


def testFcn2(x):
    """
    testFcn2
    """

    return rosen(x)


def testFcn3(x):
    return np.sum([np.power(x[i], 2) for i in range(len(x))], axis=0)
