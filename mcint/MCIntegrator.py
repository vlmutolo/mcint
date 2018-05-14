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
	of samples.

	:param xtf: Function to transform values of x tomodify the integration 
	domain.

	:param store_pts: If `True`, the `MCSimpleInt` object will save all
	evaluated points. This can run into memory constraints for large numbers
	of samples.
	"""

	def __init__(self, f, xbounds, chunk_size=1000, xtf=None, store_pts=False):
		super(MCIntegrator, self).__init__()

		# Collect user-provided information.
		self.f = f
		self.xbounds = xbounds
		self.xtf = xtf
		self.store_pts = store_pts

		# TODO: Make xbounds into an array.

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
		self.volume = np.cumprod(dim_lens)


		# Initialize some properties of the integrator.
		self.npts    = 0		# Total number of points tested
		self.f_sum   = 0		# Sum of f values
		self.fsq_sum = 0		# Sum of squares of f values

		if store_pts:
			self.eval_list = []		# List of all evaluations
		else:
			self.eval_list = None


	def update_sums(new_evals):
		self.f_sum += new_evals.sum()
		self.fsq_sum += (new_evals**2).sum()

	def calculate_estimates():
		f_mean   = f_sum   / self.npts
		fsq_mean = fsq_sum / self.npts

		int_val_est = self.volume * f_mean
		int_err_est = self.volume * np.sqrt((fsq_mean - f_mean)/self.npts)

		return (int_val_est, int_err_est)