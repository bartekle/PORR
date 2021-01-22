from solvers import chazanMiranker, gaussSeidel
from testFunctions import testFcn1, testFcn2
import numpy as np
import timeit

testFunction = testFcn1

def f(x, y): return testFunction([x, y])

initialCondition = np.array([10,10])
initCond = np.random.randint(100, size=(10))

def init_cond(n,low=-2, high=2):
	return np.random.randint(low, high, size=(n))
def init_zeros(n):
	return np.zeros(n)


def test_zb(testFunction, xsize, N, thr, eps_g, eps_mir):
	print(f"########### DANE WEJŚĆIOWE #############: \n\tfunkcja testowa: {testFunction.__name__},\n\tn: {xsize},\n\tliczba wątków: {thr}.\n")
	initCond = init_zeros(xsize)
	# initCond = np.array([10,10])
	print(f"Eps_gauss: {eps_g}, Eps_mir: {eps_mir}, Nmax: {N}")
	# print(f"Wektor wejściowy: {initCond}")

	gauss = gaussSeidel(testFunction, initCond, eps=eps_g, N=int(N))
	print(f"Algorytm Gauss-Seidel sekwencyjny: \n\t liczba iteracji: {gauss['nit']}, \n\t wartość funkcji: {gauss['fun']}, \n\t czas wykonania: {gauss['time']}\n")
	# print(f"Czas wykonania Gauss-Seidel:{gauss['time']}")
	miranker = chazanMiranker(testFunction, initCond, threads=thr, eps=eps_mir, N=int(N))
	print(f"Algorytm Chazana Mirankera równolełgy: \n\t liczba iteracji: {miranker['nit']}, \n\t wartość funkcji: {miranker['fun']}, \n\t czas wykonania: {miranker['time']}\n")
    # print(f"Czas wykonania Chazan Miranker: {miranker['time']}")

	return	gauss, miranker

# results = test_zb(testFunction=testFcn2, xsize=200, N=1e5, thr=4, eps_g=200, eps_mir=1e-1)
print(testFcn1([0,0]))
def test_przysp(testFunction, xsize, showTimes=False):
	initCond = init_cond(xsize)
	threads = [2, 4, 6, 8]
	solutions = []
	base_point = chazanMiranker(testFunction, initCond, threads=1, eps=1e-3)
	for thread in threads:
		res = chazanMiranker(testFunction, initCond, threads=thread, eps=1e-3)
		solutions.append(res)

	for idx, solution in enumerate(solutions):
		print(f"Liczba wątków: {threads[idx]}, przyśpiesznie = {base_point['time'] / solution['time']}")
	return solutions, threads

# test_przysp(Zadanie1, 20, showTimes=True)
