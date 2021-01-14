import numpy as np
from mpi4py import MPI
from testFunctions import testFcn1, testFcn2
from numpy.linalg import norm
from scipy.optimize import minimize_scalar, rosen_der
from time import perf_counter
from solvers import gaussSeidel


def chazanMirankerMPI(f, x0, N=int(1e5), eps=1e-6, solutionHistory=False, beta=1., q=0.9):
    P = x0
    fn = f(P)
    n = len(P)

    if solutionHistory:
        xHist = P.copy()
        funHist = np.array([fn])

    # calculate computing range
    numbers_per_rank = n // size
    if N % size > 0:
        numbers_per_rank += 1
    my_first = rank * numbers_per_rank
    my_last = my_first + numbers_per_rank

    initialDirections = np.eye(n)
    actualDirections = initialDirections.copy()
    for it in range(N):

        xPoints = np.tile(P, (n, 1)) + \
                  np.cumsum(actualDirections, axis=0)

        # prepare alphas array for scattering
        # musi być array, który ma *size* elementów, więc trzeba podzielić ręcznie
        if rank == 0:
            alphas = np.zeros((n,))
            alphas_prep = np.array_split(alphas, size)
        else:
            alphas_prep = None

        # scatter alphas array to all processess
        alphas = comm.scatter(alphas_prep, root=0)

        # This loop is parallelizable
        # iter temp dlatego, żeby indeks w alphas[] nie przekraczał len(n) - 1, a indeksy te nie mają znaczenia po comm.gather
        iter_temp = 0
        # iterate over each process range
        for pointInd in range(my_first, my_last):
            def dirFcn(alpha):
                arg = xPoints[pointInd] + alpha * actualDirections[0]
                return f(arg)

            res = minimize_scalar(dirFcn)
            alphas[iter_temp] = res.x
            iter_temp += 1

        # gather all chunks & form proper array
        alphas = comm.allgather(alphas)
        alphas = np.concatenate(alphas)

        # compute new point
        Pnew = xPoints[0] + alphas[0] * actualDirections[0]

        # if norm(Pnew - P) < eps:
        #     break
        # else:
        #     P = Pnew

        if norm(rosen_der(Pnew) - rosen_der(P)) < eps:
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

    def init_zeroes(n):
        return np.zeros(n)

    testFunction = testFcn2
    initialCondition = init_cond(4)
    preparedData_4 = np.array([3, 3, 9, 2])  #
    # preparedData_64 = np.ones((64))
    preparedData_64 = np.hstack([1,2,3,4]*16)  # f(chazan(preparedData_64)) = 1.121275

    def prepareData(n):
        return np.hstack([1,-2,3,-4] * (n//4))
    # initialCondition = np.array([10, 10])

    # def f(x):
    #     return testFunction(x)

    data = init_zeroes(24)

    # parallel MPI
    start_time = perf_counter()
    result = chazanMirankerMPI(testFunction, data, eps=1, N=int(1e5))
    stop_time = perf_counter()

    # sequential
    start_time_gauss = perf_counter()
    result_gauss = gaussSeidel(testFunction, data, eps=1e-3, N=int(1e5))
    stop_time_gauss = perf_counter()
    if rank == 0:
        print(f"Output computed X vector:\n {result['x']} \n")
        print(f"Wartosc funkcji MPI: {result['fun']}")
        print(f"Liczba iteracji MPI: {result['nit']}")
        print(f"Czas wykonywania (parallel): {stop_time-start_time} s \n")
        print(f"Wartosc funkcji gauss: {result_gauss['fun']}")
        print(f"Liczba iteracji gauss: {result_gauss['nit']}")
        print(f"Czas wykonywania (sekwencyjna): {stop_time_gauss-start_time_gauss} s")
