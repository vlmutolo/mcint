import numpy as np


class MCSimpleInt():
	"""
	Every instance of this class will be able to perform naïve Monte
	Carlo integration for a single function over a single set of bounds.
	To perform two or more integrals (perhaps with changing integrands, 
	or different bounds of integration), create two or more instances
	of this class.

	:param f: The function to be integrated. It *must* be vectorized. Any
	function which only performs valid numpy array operations is supported.
	Otherwise, apply the `@np.vectorize` decorator to the function.

	:param xbounds: Minimum and maximum values for each x (pre-transformation).
	`xbounds` should be an array of shape ($N$,2). It can also be an iterable of
	iterables, typically of the form [[x0_min, x0_max], [x1_min, x1_max], …].

	:param chunk_size: The integration will proceed by evaluating `chunk_size`
	number of points at a time. This is to preserve memory for large numbers
	of samples. In other words, the program will hold `chunk_size` points
	in memory at a time.

	:param xtf: Function to transform values of x to modify the integration 
	domain. Like `f`, this also must be vectorized. It takes the form
	xtf(x0, x1, x2, …) and outputs (x0_tf, x1_tf, x2_tf, …).

	:param store_pts: If `True`, the `MCSimpleInt` object will save all
	evaluated points. This can run into memory constraints for large numbers
	of samples.

	:param prng: Pseudo-random number generator. Prove `int` as a seed, a
	NumPy `RandomState` object, or None to use a random seed.
	"""

	def __init__(self, f, xbounds, chunk_size=1000, xtf=None, store_pts=False,
						prng=None, args=tuple(), kwargs=dict()):
		super(MCSimpleInt, self).__init__()

		# Collect user-provided information.
		self.f = f
		self.xtf = xtf
		self.store_pts = store_pts
		self.args = args
		self.kwargs = kwargs

		# Make xbounds into an array.
		xbounds_list = []
		for i in range(len(xbounds)):
			xbounds_list.append([ xbounds[i][0], xbounds[i][1] ])
		self.xbounds = np.array(xbounds_list)

		# Calculate transformed bounds if needed.
		if self.xtf != None:
			xbounds_tf = self.xtf(self.xbounds)
		else:
			xbounds_tf = self.xbounds

		# TODO: Check to make sure that every max is greater than every min.
		for i in range(len(self.xbounds)):
			if xbounds_tf[i,1] > xbounds_tf[i,0]:
				continue
			else:
				raise ValueError('Upper bounds must be strictly greater' + 
									' than lower bounds.')

		# Calculate volume of integration region.
		dim_lens = []
		for i in range(len(xbounds_tf)):
			length = xbounds_tf[i,1] - xbounds_tf[i,0]
			dim_lens.append(length)
		self.volume = np.cumprod(dim_lens)[-1]


		# Initialize some properties of the integrator.
		self.npts    = 0		# Total number of points tested
		self.f_sum   = 0		# Sum of f values
		self.fsq_sum = 0		# Sum of squares of f values

		if store_pts:
			self.eval_list = []		# List of all evaluations
		else:
			self.eval_list = None


		# Handle prng.
		if type(prng) == int:
			self.prng = np.RandomState(prng)
		elif type(prng) == np.random.RandomState:
			self.prng = prng
		else:
			self.prng = np.random.RandomState()



	def add_evals(self, num):

		# dom_samples is a list of arrays of samples from each domain.
		# Ex.: [x0_samps_arr, x1_samps_arr, x2_samps_arr, …]
		dom_samples = []
		for i in range(len(self.xbounds)):
			dom_range = self.xbounds[i,1] - self.xbounds[i,0]
			dom_pts = self.prng.rand(num) * dom_range + self.xbounds[i,0]
			dom_samples.append(dom_pts)

		if self.xtf != None:
			dom_tf_samples = self.xtf(*dom_samples)
		else:
			dom_tf_samples = dom_samples

		evals = self.f(*dom_tf_samples, *self.args, **self.kwargs)

		self.update_sums(evals)

		if self.store_pts:
			self.eval_list.append(evals)


	def update_sums(self, new_evals):
		new_evals_arr = np.array(new_evals)
		self.f_sum += new_evals_arr.sum()
		self.fsq_sum += (new_evals_arr**2).sum()
		self.npts += len(new_evals)

	def calculate_estimates(self):
		if self.npts == 0:
			raise ValueError('npts is 0; add_evals must be run at least once before' +
								' calculate_estimates can be used.')

		f_mean   = self.f_sum   / self.npts
		fsq_mean = self.fsq_sum / self.npts

		int_val_est = float(self.volume * f_mean)
		int_err_est = float(self.volume * np.sqrt((fsq_mean - f_mean)/self.npts))

		return (int_val_est, int_err_est)