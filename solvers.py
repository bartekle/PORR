from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from numpy.linalg import norm
import pymp
import numpy as np
from time import perf_counter


def gaussSeidel(f, x0, N=int(1e3), eps=1e-6, solutionHistory=False):
    """
    Sequential Gauss Seidel Method
    """
    start_time = perf_counter()
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
        # debug
        # print(f"Gauss: {f(xn)}, it: {n+1}")
        if (f(xn) - 0 < eps):
            break

    end_time = perf_counter()
    retval = {
        'x': xn,
        'fun': res.fun,
        'nit': n + 1,
        'time': end_time - start_time
    }

    if solutionHistory:
        retval['xHist'] = xHist
        retval['funHist'] = funHist


    return retval


def chazanMiranker(f, x0, N=int(1e3), eps=1e-6, solutionHistory=False, beta=1., q=0.9, threads=1):
    """
    Parallel Gauss Seidel Method variant
    """
    start_time = perf_counter()
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

        alphas = pymp.shared.array((n,),dtype='float32')

        # This loop is parallelizable
        with pymp.Parallel(threads) as p:
            for pointInd in p.range(0, n):
                def dirFcn(alpha):
                    arg = xPoints[pointInd] + alpha * actualDirections[0]
                    return f(arg)

                res = minimize_scalar(dirFcn)
                alphas[pointInd] = res.x

        Pnew = xPoints[0] + alphas[0] * actualDirections[0]
        # debug
        # print(f"Chazan: {f(Pnew)}, it: {it+1}")
        if f(Pnew) - 0 < eps:
            P = Pnew
            break
        else:
            if f(Pnew) < f(P):
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
    end_time = perf_counter()
    retval = {
        'x': P,
        'fun': f(P),
        'nit': it + 1,
        'time': end_time - start_time
    }

    if solutionHistory:
        retval['xHist'] = xHist
        retval['funHist'] = funHist

    return retval
