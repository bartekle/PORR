import numpy as np
from mpi4py import MPI
from testFunctions import testFcn1, testFcn2
from numpy.linalg import norm
from scipy.optimize import minimize_scalar
from time import perf_counter
from solvers import gaussSeidel


def chazanMirankerMPI(f, x0, N=int(1e3), eps=1e-6, solutionHistory=False, beta=1., q=0.9):
    P = x0
    fn = f(P)
    n = len(P)

    if solutionHistory:
        xHist = P.copy()
        funHist = np.array([fn])

    # calculate computing range
    numbers_per_rank = N // size
    if N % size > 0:
        numbers_per_rank += 1
    my_first = rank * numbers_per_rank
    my_last = my_first + numbers_per_rank

    initialDirections = np.eye(n)
    actualDirections = initialDirections.copy()

    for it in range(my_first, my_last):

        xPoints = np.tile(P, (n, 1)) + \
                  np.cumsum(actualDirections, axis=0)

        alphas = np.empty((n,), dtype='float32')
        # This loop is parallelizable
        for pointInd in range(n):
            def dirFcn(alpha):
                arg = xPoints[pointInd] + alpha * actualDirections[0]
                return f(arg)

            res = minimize_scalar(dirFcn,method='brent')
            alphas[pointInd] = res.x

        # compute new point
        Pnew = xPoints[0] + alphas[0] * actualDirections[0]

        # break if reached desired precision
        if norm(Pnew - P) < eps:
            break
        else:
            P = Pnew

        if solutionHistory:
            xHist = np.vstack((xHist, P))
            funHist = np.vstack((funHist, f(P)))

        # modify directions base
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

    for i in retval:
        print(i, retval[i])

    if solutionHistory:
        retval['xHist'] = xHist
        retval['funHist'] = funHist

    return retval


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    def init_cond(n, range=100):
        return np.random.randint(range, size=(n))

    testFunction = testFcn1
    initialCondition = init_cond(4)
    preparedData_4 = np.array([3, 3, 9, 2])  #
    # preparedData_64 = np.ones((64))
    preparedData_64 = np.hstack([1,2,3,4]*16)  # f(chazan(preparedData_64)) = 1.121275

    def prepareData(n):
        return np.hstack([1,2,3,4] * (n//4))
    # initialCondition = np.array([10, 10])

    def f(x):
        return testFunction(x)

    data = prepareData(32)

    # parallel MPI
    start_time = perf_counter()
    result = chazanMirankerMPI(testFunction, data)
    stop_time = perf_counter()

    # sequential
    start_time_gauss = perf_counter()
    result_gauss = gaussSeidel(testFunction, data)
    stop_time_gauss = perf_counter()
    # if rank == 0:
    #     print(f"Output computed X vector:\n {result['x']}")
    #     print(f"Wartosc funkcji: {result['fun']}")
    #     print(f"Czas wykonywania (parallel): {stop_time-start_time} s")
    #     print(f"Wartosc funkcji gauss: {result_gauss['fun']}")
    #     print(f"Czas wykonywania (sekwencyjna): {stop_time_gauss-start_time_gauss} s")
