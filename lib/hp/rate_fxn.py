"""
Allows us to manage the dynamic rate function of a Hawkes Process
"""

# python imports
import numpy as np
import numpy.random as npr

class HawkesProcessRate():

    def __init__(self,base,alpha,beta):
        self.hist = []
        self.base = base
        self.alpha = alpha
        self.beta = beta

    def __call__(self,time):
        # return rate at time
        rate = self.base(time)
        for event in self.hist:
            t,k = event # time and mark
            delta_t = time - t
            if delta_t < 0: break
            offspring_intensity = self.alpha(k)
            offspring_intensity *= self.beta(delta_t,k)
            rate += offspring_intensity
        return rate

    def update(self,event):
        self.hist.append(event)

    def set_hist(self,history):
        self.hist = history

    def reset(self):
        self.hist = []
