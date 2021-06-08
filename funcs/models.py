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
        self.T_out = np.array(T_out)
        self.Q_in  = Q_in

        # If only given parameters for one branch, recast shape of arrays (n_seg) -> (1, n_seg)
        if len(self.T.shape) == 1:
            self.k     = self.k[np.newaxis, :]
            self.c     = self.c[np.newaxis, :]
            self.T     = self.T[np.newaxis, :]
            self.T_out = self.T_out.reshape(1)
            
        n_br, n_seg = self.T.shape
        self.n_br  = n_br # Number of branches
        self.n_seg = n_seg # Number of segments per branch 
        # NOTE currently all branches must have same no. of segments, need to change this (or have empty segments to ensure same shape)
        
        assert (len(self.k) == n_br) & (len(self.c) == n_br), 'input parameter shape mismatch'
        
    def update(self, timestep):
        """
        Calculate heat flux between segments and update temperatures

        Parameters
        ----------
        timestep : float
            Time interval between iterations
        Q_in : float
            Updated value of input heat 
        """
        heat_flux = np.diff(np.hstack((self.T,self.T_out[:,np.newaxis])))*self.c
        net_heat_flux = np.diff(heat_flux)
        
        # update non central nodes
        self.T[:, 1:] += ( net_heat_flux / self.k[:,1:] ) * timestep
        
        # update central node
        self.T[:, 0] += ( (heat_flux[:,0] + self.Q_in) / self.k[:,0] ).sum() * timestep 
        
    def run(self, iterations, timestep, plot=True):
        """
        Run simulation for given number of iterations with given timestep

        Parameters
        ----------
        iterations : int
            number of iterations to run the simulation
        
        timestep : float
            time interval between each iteration
        
        plot : bool, default = True
            Set to False to omit plot

        Returns
        -------
        Ts : ndarray, shape (number of iterations, number of segments)
            numpy array of temperatures for each segment, for each timestep
        """
        Ts = np.empty(shape=(iterations, self.n_br, self.n_seg))
        Qs = []
        for i in range(iterations):
#             self.Q_in = 0.5 * ( 1 + 0.6*np.sin(2*np.pi*i/50.0)) # generalise this to any function or input data
            self.update(timestep)
            Ts[i] = self.T
            Qs.append(self.Q_in)
            
        Ts = np.transpose(Ts,axes=(1,2,0)) # New shape has (n_branches, n_segments, n_iterations)
        
        if plot:
            
            time = np.linspace(0,iterations*timestep/(3600*24),iterations, endpoint=False)
            fig, ax = plt.subplots(1,1, figsize=(20,10))
            for i, T_br in enumerate(Ts):
                for j, T_seg in enumerate(T_br[1:]):
                    ax.plot(time, T_seg, label='branch: {}, segment: {}'.format(i,j))
            ax.plot(time, Ts[0,0,:], label='Central temperature')
            ax.set(xlabel='Time (days)', ylabel='Temperature (C)')
            ax.legend()
            
        return Ts, Qs