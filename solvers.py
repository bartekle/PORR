from scipy.optimize import minimize_scalar
from numpy.linalg import norm
import numpy as np


def gaussSeidel(f, x0, N=int(1e5), eps=1e-6, solutionHistory=False):
    """
    Sequential Gauss Seidel Method
    """
    xn = x0
    fn = f(x0)

    if solutionHistory:
        xHist = xn.copy()
        funHist = np.array([fn])

    for n in range(N):
        xlast = xn.copy()
        for i in range(len(xn)):
            def dirFcn(x):
                xInternal = xn.copy()
                xInternal[i] = x
                return f(xInternal)

            res = minimize_scalar(dirFcn)
            xn[i] = res.x

        if solutionHistory:
            xHist = np.vstack((xHist, xn))
            funHist = np.vstack((funHist, res.fun))

        if (norm(xn - xlast) < eps):
            break

    retval = {
        'x': xn,
        'fun': res.fun,
        'nit': n + 1
    }

    if solutionHistory:
        retval['xHist'] = xHist
        retval['funHist'] = funHist

    return retval


def chazanMiranker(f, x0, N=int(1e5), eps=1e-6, solutionHistory=False, beta=1., q=0.9):
    """
    Parallel Gauss Seidel Method variant
    """
    P = x0
    fn = f(P)
    n = len(P)

    if solutionHistory:
        xHist = P.copy()
        funHist = np.array([fn])

    initialDirections = np.eye(n)
    actualDirections = initialDirections.copy()

    for it in range(N):
        xPoints = np.tile(P, (n, 1)) + \
                  np.cumsum(actualDirections, axis=0)

        alphas = np.empty((n,), dtype='float32')
        # This loop is parallelizable
        for pointInd in range(n):
            def dirFcn(alpha):
                arg = xPoints[pointInd] + alpha * actualDirections[0]
                return f(arg)

            res = minimize_scalar(dirFcn)
            alphas[pointInd] = res.x
            print(f"Res is {res.x}")

        Pnew = xPoints[0] + alphas[0] * actualDirections[0]

        if (norm(Pnew - P) < eps):
            break
        else:
            P = Pnew

        if solutionHistory:
            xHist = np.vstack((xHist, P))
            funHist = np.vstack((funHist, f(P)))

        for dirInd in range(n - 1):
            actualDirections[dirInd] = actualDirections[dirInd + 1] + \
                                       (alphas[dirInd + 1] - alphas[dirInd]) * \
                                       actualDirections[0]

        newDir = beta * initialDirections[it % initialDirections.shape[-1]]
        beta *= q

        actualDirections[-1] = newDir


    retval = {
        'x': P,
        'fun': f(P),
        'nit': it + 1
    }

    if solutionHistory:
        retval['xHist'] = xHist
        retval['funHist'] = funHist

    return retval
