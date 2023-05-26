import numpy as np


# Implementaion of the Orstein Urlhembeck noise perturbation algorithm
class OUNoise:
    def __init__(self, mean, std_diviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_diviation = std_diviation
        self.dt = dt
        self.x_initial = x_initial
        
        self.reset()
    
    
    def __call__(self):
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.std_diviation * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        # after performing the perturbation set the new x values to the previous x values
        self.x_prev = x # this ensures that the next noise is dependent on the current one
        # return the x values
        return x

        
        
    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            # does this mean that the mean provided should be an array?
            self.x_prev = np.zeros_like(self.mean)