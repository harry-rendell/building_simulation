import numpy as np
import matplotlib.pyplot as plt

class model_ND():
	def __init__(self, name, k, c, T, T_out, Q_in):
		"""
		Initialise class and set system parameters

		Parameters
		---------
		name : str
			Name of model, used in axis legend
		k : array of floats
			Thermal conductivity of segments (W m^-1 K^-1)
		c : array of floats
			Heat capacity of segments (J K^-1)
		T : array of floats
			Initial temperature of segments (Celcius)
		T_out : array of floats
			Data for outside temperature (Celcius)
		Q_in : float
			 Heat supply to central node (W)
		"""
		self.name  = name
		self.k	 = np.array(k) # Thermal conductivity of segments
		self.c	 = np.array(c) # Heat capacity of segments
		self.T	 = np.array(T) # Temperatures of segments
		self.T_out = np.expand_dims(T_out, axis=-1)
		self.Q_in  = Q_in

		if len(self.T.shape) == 0:
			n_seg = 1
			n_br  = 1
		elif len(self.T.shape) == 1:
			n_seg = self.T.shape[0]
			n_br  = 1
		else:
			n_br, n_seg = self.T.shape

		# If class is only given parameters for a single segment (float), or for a single branch (array), recast arrays
		if len(self.T.shape) < 2:
			shape	  = (1,-1) # Calculate the appropriate shape to cast to
			self.k	 = self.k.reshape(shape)
			self.c	 = self.c.reshape(shape)
			self.T	 = self.T.reshape(shape)
			self.T_out = np.expand_dims(self.T_out, axis=-1)

		self.n_br  = n_br # Number of branches
		self.n_seg = n_seg # Number of segments per branch
		# NOTE currently all branches must have same no. of segments

		assert (len(self.k) == n_br) & (len(self.c) == n_br), 'input parameter shape mismatch'

	def update(self, i, dt):
		"""
		Calculate heat flux between segments and update temperatures

		Parameters
		----------
		i  : int
			counter to select appropriate T_out and Q_in values
		dt : float
			Time interval to update new temperatures
		"""
		heat_flux = np.diff(np.hstack((self.T,self.T_out[i])))*self.k
		net_heat_flux = np.diff(heat_flux)

		# update non central nodes
		self.T[:, 1:] += ( net_heat_flux / self.c[:,1:] ) * dt

		# update central node
		self.T[:, 0] += ( (heat_flux[:,0] + self.Q_in[i]) / self.c[:,0] ).sum() * dt

	def run(self, times):
		"""
		Run simulation for given number of iterations with given time values

		Parameters
		----------
		times : array
			set of time values to run the simulation
			
		Returns
		-------
		Ts : ndarray, shape (number of iterations, number of segments)
			numpy array of temperatures for each segment, for each time step
		"""
		Ts = np.empty(shape=(len(times), self.n_br, self.n_seg))
		Ts[0] = self.T
		self.times = times
		dtimes = np.diff(times)
		for i, t in enumerate(times[1:]):
			self.update(i, dtimes[i])
			Ts[i+1] = self.T

		Ts = np.transpose(Ts,axes=(1,2,0)) # New shape has (n_branches, n_segments, n_iterations)
		self.Ts = Ts

		return Ts

	def plot(self, figax=None, show_heating=True, **kwargs):
		"""
		Plot the simulation

		Parameters
		----------
		figax : tuple
			tuple of (figure, axis handle). If None, generate new (fig, ax)
		show_heating : bool
			True to show the heating profile on the same plot
		kwargs
			extra keyword arguments to pass to ax.plot

		Returns
		-------
		fig : figure
		ax  : axis handle
		"""
		cmap = plt.get_cmap('jet')
		# Create new figure and axes handles if none supplied
		if figax == None:
			fig, ax = plt.subplots(1,1, figsize=(15,6))
		else:
			fig, ax = figax

		for i, T_br in enumerate(self.Ts):
			for j, T_seg in enumerate(T_br[1:]):
				ax.plot(self.times/3600, T_seg, label=self.name+': branch: {}, segment: {}'.format(i,j), color=cmap(0.8*i/self.n_br + 0.2*j/self.n_seg), **kwargs)
		ax.plot(self.times/3600, self.Ts[0,0,:], label=self.name+': Central temperature', color='k', **kwargs)
		ax.set(xlabel='Time (hours)', ylabel='Temperature (C)')
		ax.legend()

		if show_heating:
			ax2 = ax.twinx()
			ax2.plot(self.times/3600, self.Q_in, ls='--', lw=1, color='r', label='Heat input')
			ax2.set(ylabel='Heat input (W)')
			ax2.legend(loc=7)

		return fig, ax
