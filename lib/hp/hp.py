import pickle,os
from os import path as osp
import _init_paths
from base_utils import *
import pandas as pd
import numpy as np
import numpy.random as npr
import mnist_loader
from distributions import Gamma
from distributions import WeibullDistribution as Weibull
from distributions import MultinomialDistribution as Multinomial
from processes import PoissonProcess
import matplotlib.pyplot as plt


def const_rate(tau):
    return 2*tau
def const_hazard(tau):
    return npr.uniform(0,tau)
def weibull_rate(tau,shape=1,scale=1):
    mean = (tau/scale)**shape
    return const*tau
def weibull_hazard(tau,shape=1,scale=1):
    rv = Weibull({'shape':shape,'scale':scale},
                 is_hazard_rate=True)
    return rv

def gen_next_sk(lamTk,a,delta):
    u1 = npr.uniform(0,1)
    u2 = npr.uniform(0,1)
    D_k1 = 1 + delta * np_loge(u1) / (lamTk - a)
    S1_k1 = np.inf
    if D_k1 > 0:
        S1_k1 = -1/delta * np_loge(D_k1)
    S2_k2 = -1/a * np_loge(u2)
    S_k = np.min([S1_k1,S2_k2])
    return S_k

def gen_next_lam(lamTk,a,delta,Sk1):
    lamTk1_minus = (lamTk - a) * np.exp(-delta * Sk1) + a
    return lamTk1_minus

def generate_hawkes_process(G,a=.9,delta=1.0,lam0=0.9,time_f=100):
    
    # init first event time
    S_1 = gen_next_sk(lam0,a,delta)
    lamTk_minus = gen_next_lam(lam0,a,delta,S_1)
    mark = G.sample(1)[0]
    lamTk = lamTk_minus + mark

    events = [S_1]
    marks = [mark]
    lams = [lamTk]

    # get all the rest
    while (events[-1] < time_f):
        # sample inter-event time
        S_k = gen_next_sk(lamTk,a,delta)
        events.append(events[-1] + S_k)

        # sample mark
        mark = G.sample(1)[0]
        marks.append(mark)

        # update rate constant
        lamTk_minus = gen_next_lam(lamTk,a,delta,S_k)
        lamTk = lamTk_minus + mark
        lams.append(lamTk)

    return events,marks,lams


if __name__ == "__main__":
    print("HI")
