def VecFnc(input, n = 4,
			a1 = 10.0,
			a2 = 10.0,
			Kdxx = 4,
			Kdyx = 4,
			Kdyy = 4,
			Kdxy = 4,
			b1 = 10.0,
			b2 = 10.0,
			k1 = 1.0,
			k2 = 1.0,
			c1 = 0):
	x, y = input
	dxdt = c1 + a1*(x**n)/(Kdxx**n + (x**n)) + (b1*(Kdyx**n))/(Kdyx**n + (y**n)) - (x*k1)
	dydt = c1 + a2*(y**n)/(Kdyy**n + (y**n)) + (b2*(Kdxy**n))/(Kdxy**n + (x**n)) - (y*k2)

	return [dxdt, dydt]

numPaths, numTimeSteps, pot_path, path_tag, attractors_pot, x_path, y_path = path_integral(VecFnc, x_lim=[0, 40], y_lim=[0, 40], xyGridSpacing=2, dt=1e-2, tol=1e-2, numTimeSteps=1400)
Xgrid, Ygrid, Zgrid = alignment(numPaths, numTimeSteps, pot_path, path_tag, attractors_pot, x_path, y_path)
