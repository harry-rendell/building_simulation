import numpy as np

def loss(params, method, times, Ts_true, T_out, Q_in):
	"""
	Loss function to pass to optimiser. Note that this only currently works with a 2 segment set-up.
	Parameters
	----------
	params : list of floats
			list of [k1, k2, c1, c2, T1, T2]
	method : str
		choose a method to calculate loss.
		Possible choices:
			mse - mean square error
			mad - mean absolute deviation
			lae - least absolute error
			sse - sum of squares
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

		heat_flux = np.diff(np.hstack((T_est,T_out)))*k_est
# 		heat_flux = np.diff(np.hstack((T_est,T_out[i])))*k_est
		net_heat_flux = np.diff(heat_flux)

		# update non central nodes
		T_est[:, 1:] += ( net_heat_flux / c_est[:,1:] ) * dtimes[i]

		# update central node
		T_est[:, 0] += ( (heat_flux[:,0] + Q_in) / c_est[:,0] ).sum() * dtimes[i]
# 		T_est[:, 0] += ( (heat_flux[:,0] + Q_in[i]) / c_est[:,0] ).sum() * dtimes[i]

		Ts[i+1] = T_est

	Ts = np.transpose(Ts,axes=(1,2,0)) # New shape has (n_branches, n_segments, n_iterations)

	if method=='mse':
		# mean square error
		error = np.mean((Ts_true - Ts) ** 2)
	elif method=='mad':
		# mean absolute deviation
		error = np.mean(abs(Ts_true-Ts))
	elif method=='lae':
		# least absolute error
		error = np.sum(abs(Ts_true-Ts))
	elif method=='sse':
		# sum of squares error
		error = np.sum((Ts_true - Ts) ** 2)
	else:
		raise ValueError('Method not supported')

	return error

def loss_single_seg(params, method, times, Ts_true, T_out, Q_in):
	"""
	Loss function to pass to optimiser. Note that this only currently works with a 1 segment set-up.
	Parameters
	----------
	params : list of floats
			list of [k, c, T]
	method : str
		choose a method to calculate loss.
		Possible choices:
			mse - mean square error
			mad - mean absolute deviation
			lae - least absolute error
			sse - sum of squares
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

# 	Ts = np.transpose(Ts,axes=(1,2,0)) # New shape has (n_branches, n_segments, n_iterations)

	if method=='mse':
		# mean square error
		error = np.mean((Ts_true - Ts) ** 2)
	elif method=='mad':
		# mean absolute deviation
		error = np.mean(abs(Ts_true-Ts))
	elif method=='lae':
		# least absolute error
		error = np.sum(abs(Ts_true-Ts))
	elif method=='sse':
		# sum of squares error
		error = np.sum((Ts_true - Ts) ** 2)
	else:
		raise ValueError('Method not supported')

	return error

def loss_single_seg_fixed_T(params, method, times, T0, Ts_true, T_out, Q_in):
	"""
	Loss function to pass to optimiser. Note that this only currently works with a 1 segment set-up.
	Parameters
	----------
	params : list of floats
			list of [k, c]
	method : str
		choose a method to calculate loss.
		Possible choices:
			mse - mean square error
			mad - mean absolute deviation
			lae - least absolute error
			sse - sum of squares
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

# 	Ts = np.transpose(Ts,axes=(1,2,0)) # New shape has (n_branches, n_segments, n_iterations)

	if method=='mse':
		# mean square error
		error = np.mean((Ts_true - Ts) ** 2)
	elif method=='mad':
		# mean absolute deviation
		error = np.mean(abs(Ts_true-Ts))
	elif method=='lae':
		# least absolute error
		error = np.sum(abs(Ts_true-Ts))
	elif method=='sse':
		# sum of squares error
		error = np.sum((Ts_true - Ts) ** 2)
	else:
		raise ValueError('Method not supported')

	return error

def loss_two_seg(params, method, times, T0, Ts_true, T_out, Q_in):
	"""
	Loss function to pass to optimiser. Note that this only currently works with a 2 segment set-up.
	Parameters
	----------
	params : list of floats
			list of [k, c]
	method : str
		choose a method to calculate loss.
		Possible choices:
			mse - mean square error
			mad - mean absolute deviation
			lae - least absolute error
			sse - sum of squares
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
		
# 	Ts = np.transpose(Ts,axes=(1,2,0)) # New shape has (n_branches, n_segments, n_iterations)

	if method=='mse':
		# mean square error
		error = np.mean((Ts_true - Ts) ** 2)
	elif method=='mad':
		# mean absolute deviation
		error = np.mean(abs(Ts_true-Ts))
	elif method=='lae':
		# least absolute error
		error = np.sum(abs(Ts_true-Ts))
	elif method=='sse':
		# sum of squares error
		error = np.sum((Ts_true - Ts) ** 2)
	else:
		raise ValueError('Method not supported')

	return error