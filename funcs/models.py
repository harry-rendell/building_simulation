import numpy as np
import matplotlib.pyplot as plt

class model_ND():
    def __init__(self, k, c, T, T_out, Q_in):
        """
        Initialise class and set system parameters

        Parameters
        ----------
        k : array of floats
            Thermal conductivity of segments
        c : array of floats
            Heat capacity of segments
        T : array of floats
            Initial temperature of segments
        T_out : float
            Outside temperature (Celcius)
        Q_in : float
             Heat supply to segment 0 (Watts)
        """
        self.k     = np.array(k) # Thermal conductivity of segments
        self.c     = np.array(c) # Heat capacity of segments
        self.T     = np.array(T) # Temperatures of segments
        self.T_out = np.array(T_out).reshape(-1,1)
        self.Q_in  = Q_in
        self.T_out = self.T_out
        
        # These conditional statements are a little convoluted, find a better way of setting n_br, n_seg
        if len(self.T.shape) == 0:
            n_seg = 1
            n_br  = 1
        elif len(self.T.shape) == 1:
            n_seg = self.T.shape[0]
            n_br  = 1
        else:
            n_br, n_seg = self.T.shape
            
        # If only given parameters for a single segment (number), or for a single branch (vector), recast arrays
        if len(self.T.shape) < 2:
            shape      = (1,-1) # Calculate the appropriate shape to cast to
            self.k     = self.k.reshape(shape)
            self.c     = self.c.reshape(shape)
            self.T     = self.T.reshape(shape)
        
        self.n_br  = n_br # Number of branches
        self.n_seg = n_seg # Number of segments per branch 
        # NOTE currently all branches must have same no. of segments, need to change this (or have empty segments to ensure same shape)
        
        assert (len(self.k) == n_br) & (len(self.c) == n_br), 'input parameter shape mismatch'
        
    def update(self, dt):
        """
        Calculate heat flux between segments and update temperatures

        Parameters
        ----------
        dt : float
            Time interval to update new temperatures to 
        Q_in : float
            Updated value of input heat 
        """
        heat_flux = np.diff(np.hstack((self.T,self.T_out)))*self.c
        net_heat_flux = np.diff(heat_flux)
        
        # update non central nodes
        self.T[:, 1:] += ( net_heat_flux / self.k[:,1:] ) * dt
        
        # update central node
        self.T[:, 0] += ( (heat_flux[:,0] + self.Q_in) / self.k[:,0] ).sum() * dt
        
    def run(self, times, plot=True):
        """
        Run simulation for given number of iterations with given time values

        Parameters
        ----------
        iterations : int
            number of iterations to run the simulation
        
        times : array
            set of time values to run the simulation
        
        plot : bool, default = True
            Set to False to omit plot

        Returns
        -------
        Ts : ndarray, shape (number of iterations, number of segments)
            numpy array of temperatures for each segment, for each time step
        """
        Ts = np.empty(shape=(len(times), self.n_br, self.n_seg))
        Ts[0] = self.T
        Qs = []
        dtimes = np.diff(times)
        for i in range(len(times)-1):
#             self.Q_in = 0.5 * ( 1 + 0.6*np.sin(2*np.pi*i/50.0)) # generalise this to any function or input data
            self.update(dtimes[i])
            Ts[i+1] = self.T
            Qs.append(self.Q_in)
            
        Ts = np.transpose(Ts,axes=(1,2,0)) # New shape has (n_branches, n_segments, n_iterations)
        
        if plot:
            
            fig, ax = plt.subplots(1,1, figsize=(20,10))
            for i, T_br in enumerate(Ts):
                for j, T_seg in enumerate(T_br[1:]):
                    ax.plot(times/3600, T_seg, label='branch: {}, segment: {}'.format(i,j))
            ax.plot(times/3600, Ts[0,0,:], label='Central temperature')
            ax.set(xlabel='Time (hours)', ylabel='Temperature (C)')
            ax.legend()
            
        return Ts, Qs