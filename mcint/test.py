import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from time import time
from importlib import reload
import MCIntegrator
reload(MCIntegrator)

@np.vectorize
def f(x):
	return x**2

def g(x, y):
	return x**2 + y**2

start = time()
runs = 10

mci = MCIntegrator.MCSimpleInt(f, [[0,2]])
mci.add_evals(runs)
mci_res  = mci.calculate_estimates()
end0 = time()

mci = MCIntegrator.MCSimpleInt(g, [[0,2], [0,2]])
mci.add_evals(runs)
mci_res  = mci.calculate_estimates()
end1 = time()

print((end1 - start) / (end0 - start))