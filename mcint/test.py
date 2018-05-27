import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from time import time
from importlib import reload
import MCIntegrator
reload(MCIntegrator)




runs_domain = np.linspace(1, 1000, 10, dtype=np.int)
args_domain = np.linspace(1,   10,  3, dtype=np.int)
time_lists = []

for i in range(1, 5):

	times = []

	# Define a function to take however many arguments,
	# square them, and then return the sum.
	def f(*args):
		res = 0
		for j in range(len(args)):
			res += args[j]**2
		return res

	# We will set the bounds to (0,1) for each
	# variable of integration.
	bounds = [[0,1]]*i

	for num_runs in runs_domain:

		start = time()
		mci = MCIntegrator.MCSimpleInt(f, bounds)
		mci.add_evals(num_runs)
		mci_res  = mci.calculate_estimates()
		end = time()

		times.append(end - start)

	time_lists.append(times)


