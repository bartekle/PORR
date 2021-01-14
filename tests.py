from matplotlib import cm, ticker
import matplotlib.pyplot as plt
from solvers import chazanMiranker
from testFunctions import testFcn1, testFcn2
import numpy as np

solver = chazanMiranker
testFunction = testFcn2
initialCondition = np.array([10,10])
iterations = 100
collectSolutionHistory = True

s = solver(testFunction, initialCondition, N=iterations,
                solutionHistory=collectSolutionHistory)

def f(x, y): return testFunction([x, y])


fig, ax = plt.subplots(constrained_layout=True)

x1 = np.amin(s['xHist'][:, 0])
x2 = np.amax(s['xHist'][:, 0])
y1 = np.amin(s['xHist'][:, 1])
y2 = np.amax(s['xHist'][:, 1])

xInt = x2 - x1
yInt = y2 - y1

margin = 0.1

X = np.linspace(x1 - margin * xInt, x2 + margin * xInt, 100)
Y = np.linspace(y1 - margin * yInt, y2 + margin * yInt, 100)
X, Y = np.meshgrid(X, Y)


Z = f(X, Y)

CS = ax.contourf(X, Y, Z, 20)

x = s['xHist'][:, 0]
y = s['xHist'][:, 1]

ax.plot(x, y, 'ko', mfc='none')

ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1],
          scale_units='xy', angles='xy', scale=1, width=0.002, headwidth=6, headlength=10, headaxislength=7)

plt.savefig('fig.png')
