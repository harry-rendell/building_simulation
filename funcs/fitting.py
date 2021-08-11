import numpy as np

def error_method(method, x, y):
	"""
	Calculates error between x and y

	Parameters
	----------
	method : str
		method to be used to calculate loss
		Possible choices:
			mse - mean square error
			mad - mean absolute deviation
			lae - least absolute error
			sse - sum of squares
	x : array
		model data
	y : array
		true data
	
	Returns
	-------
	error : float
		error between x and y using method specified
	"""
	if method=='mse':
		# mean square error
		error = np.mean((x - y) ** 2)
	elif method=='mad':
		# mean absolute deviation
		error = np.mean(abs(x - y))
	elif method=='lae':
		# least absolute error
		error = np.sum(abs(x - y))
	elif method=='sse':
		# sum of squares error
		error = np.sum((x - y) ** 2)
	else:
		raise ValueError('Method not supported')
	
	return error
	

def loss(params, method, times, Ts_true, T_out, Q_in):
	"""
	Most general loss function to pass to optimiser.
	
	Parameters
	----------
	params : list of floats
			list containing [k1, k2, c1, c2, T1, T2]
	method : str
		choose a method to calculate loss, described in error_method
	times : array
		array of time values to calculate loss over
	Ts_true : array
		True temperatures given by data
	T_out : float
		Outside temperature
	Q_in : float
		Heat input (boiler)
		
	Returns
	-------
	error : float
			loss using method given
	"""
	
	k1, k2, c1, c2, T1, T2 = params

	k_est = np.array([k1,k2]).reshape(1,-1)
	c_est = np.array([c1,c2]).reshape(1,-1)
	T_est = np.array([T1,T2]).reshape(1,-1)

	T_out = np.array(T_out).reshape(-1,1)
	Ts = np.empty(shape=(len(times), 1, 2))
	Ts[0] = T_est

	dtimes = np.diff(times)
	for i in range(len(times)-1):

		heat_flux = np.diff(np.hstack((T_est,T_out[i, np.newaxis])))*k_est
		net_heat_flux = np.diff(heat_flux)

		# update non central nodes
		T_est[:, 1:] += ( net_heat_flux / c_est[:,1:] ) * dtimes[i]

		# update central node
		T_est[:, 0] += ( (heat_flux[:,0] + Q_in[i]) / c_est[:,0] ).sum() * dtimes[i]

		Ts[i+1] = T_est

	Ts = np.transpose(Ts,axes=(1,2,0)) # New shape has (n_branches, n_segments, n_iterations)

	error = error_method(method, Ts, Ts_true)

	return error

def loss_single_seg(params, method, times, Ts_true, T_out, Q_in):
	"""
	Loss function to pass to optimiser. This only works with a single segment set up. Temperature is allowed to vary to improve fit.
	
	Parameters
	----------
	params : list of floats
			list of [k, c, T]
	method : str
		choose a method to calculate loss, described in error_method
	times : array
		array of time values to calculate loss over
	Ts_true : array
		True temperatures given by data
	T_out : array
		Outside temperature data
	Q_in : array
		Heat input data (boiler)

	Returns
	-------
	error : float
			loss using method given
	"""
	k, c, T_est = params
	
	Ts = np.empty(shape=(len(times)))
	Ts[0] = T_est
				  
	dtimes = np.diff(times)
	
	for i in range(len(times)-1):
		dT_est = ( Q_in[i] + k*(T_out[i] - Ts[i]) )/c * dtimes[i]
		Ts[i+1] = Ts[i] + dT_est

# 	If we need to transpose Ts, use snippet below
# 	Ts = np.transpose(Ts,axes=(1,2,0)) # New shape has (n_branches, n_segments, n_iterations)

	error = error_method(method, Ts, Ts_true)

	return error

def loss_single_seg_fixed_T(params, method, times, T0, Ts_true, T_out, Q_in):
	"""
	Loss function to pass to optimiser. This only works with a single segment set-up, and a fixed initial temperature.
	
	Parameters
	----------
	params : list of floats
			list of [k, c]
	method : str
		choose a method to calculate loss, described in error_method
	times : array
		array of time values to calculate loss over
	Ts_true : array
		True temperatures given by data
	T_out : array
		Outside temperature data
	Q_in : array
		Heat input data (boiler)

	Returns
	-------
	error : float
			loss using method given
	"""
	k, c = params
	
	Ts = np.empty(shape=(len(times)))
	Ts[0] = T0
				
	dtimes = np.diff(times)
	
	for i in range(len(times)-1):
		dT_est = ( Q_in[i] + k*(T_out[i] - Ts[i]) )/c * dtimes[i]
		Ts[i+1] = Ts[i] + dT_est

	error = error_method(method, Ts, Ts_true)
# 	error = np.mean((Ts - Ts_true) ** 2) # Uncomment if error_method is not working
	
	return error

def loss_two_seg_fixed_T(params, method, times, T0, Ts_true, T_out, Q_in):
	"""
	Loss function to pass to optimiser. Note that this only currently works with a 2 segment set-up, and a fixed initial temperature.
	
	Parameters
	----------
	params : list of floats
			list of [k, c]
	method : str
		choose a method to calculate loss, described in error_method
	times : array
		array of time values to calculate loss over
	Ts_true : array
		True temperatures given by data
	T_out : array
		Outside temperature data
	Q_in : array
		Heat input data (boiler)

	Returns
	-------
	error : float
			loss using method given
	"""
	k1, k2, c1, c2 = params
	
	Ts = np.empty(shape=(len(times),2))
	Ts[0] = T0
 				  
	dtimes = np.diff(times)
	for i in range(len(times)-1):
		
		dT1 = ( Q_in[i]                 + k1*(Ts[i,1] - Ts[i,0]) )/c1 * dtimes[i]
		dT2 = ( k2*(T_out[i] - Ts[i,1]) + k1*(Ts[i,0] - Ts[i,1]) )/c2 * dtimes[i]
		
		Ts[i+1, 0] = Ts[i,0] + dT1
		Ts[i+1, 1] = Ts[i,1] + dT2
		
	error = error_method(method, Ts, Ts_true)

	return error