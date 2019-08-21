import dynamo as dyn
import numpy as np
import scipy.io

def VecFnc(input, n=4,
			a1=10.0,
			a2=10.0,
			Kdxx=4,
			Kdyx=4,
			Kdyy=4,
			Kdxy=4,
			b1=10.0,
			b2=10.0,
			k1=1.0,
			k2=1.0,
			c1=0):
	x, y=input
	dxdt=c1 + a1*(x**n)/(Kdxx**n + (x**n)) + (b1*(Kdyx**n))/(Kdyx**n + (y**n)) - (x*k1)
	dydt=c1 + a2*(y**n)/(Kdyy**n + (y**n)) + (b2*(Kdxy**n))/(Kdxy**n + (x**n)) - (y*k2)

	return [dxdt, dydt]

def test_Bhattacharya():
	""" Test the test_Bhattacharya method for mapping quasi-potential landscape.
	The original system (VecFnc) from the Bhattacharya paper and the reconstructed vector field function in the neuron
	datasets are used for testing.

	Reference: A deterministic map of Waddington’s epigenetic landscape for cell fate specification
	Sudin Bhattacharya, Qiang Zhang and Melvin E. Andersen

	Returns
	-------
	a matplotlib plot
	"""

	# simulation model from the original study
	attractors_num_X_Y, sepx_old_new_pathNum,  numPaths_att, num_attractors, numPaths, numTimeSteps, pot_path, path_tag, \
		attractors_pot, x_path, y_path = dyn.tl.path_integral(VecFnc, x_lim=[0, 40], y_lim=[0, 40], xyGridSpacing=2, dt=1e-2, tol=1e-2, numTimeSteps=1400)
	Xgrid, Ygrid, Zgrid = dyn.tl.alignment(numPaths, numTimeSteps, pot_path, path_tag, attractors_pot, x_path, y_path)

	dyn.pl.show_landscape(Xgrid, Ygrid, Zgrid)

	# neuron model
	VecFld = scipy.io.loadmat('/Volumes/xqiu/proj/dynamo/data/VecFld.mat') # file is downloadable here: https://www.dropbox.com/s/02xwwfo5v33tj70/VecFld.mat?dl=1

	def vector_field_function(x, VecFld=VecFld):
		"""Learn an analytical function of vector field from sparse single cell samples on the entire space robustly.

		Reference: Regularized vector field learning with sparse approximation for mismatch removal, Ma, Jiayi, etc. al, Pattern Recognition
		"""

		x=np.array(x).reshape((1, -1))
		if(len(x.shape) == 1):
			x = x[None, :]
		K= dyn.tl.con_K(x, VecFld['X'], VecFld['beta'])
		K = K.dot(VecFld['C'])
		return K.T

	attractors_num_X_Y, sepx_old_new_pathNum,  numPaths_att, num_attractors, numPaths, numTimeSteps, pot_path, path_tag, \
		attractors_pot, x_path, y_path = dyn.tl.path_integral(vector_field_function, x_lim=[-30, 30], y_lim=[-30, 30], xyGridSpacing=2, dt=1e-2, tol=1e-2, numTimeSteps=1400)
	Xgrid, Ygrid, Zgrid = dyn.tl.alignment(numPaths, numTimeSteps, pot_path, path_tag, attractors_pot, x_path, y_path)

	dyn.pl.show_landscape(Xgrid, Ygrid, Zgrid)
