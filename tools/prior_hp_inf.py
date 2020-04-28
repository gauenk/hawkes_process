"""
This tools runs various Hawkes Process Simulations
"""

# python imports
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pprint

# project imports
import _init_paths
import python_modules.processes as processes

def update_hp(lam_t,thinned,V):
    """
    dynamically change the hawkes process 
    rate_function as we observe more
    of the trajectory
    """
    return None

def thinning(V,lam_ub,lam_t):
    # lam_ub :: the generating, upper-bound lambda rate
    # lam_t :: the target lambda rate
    thinned = np.zeros(len(V)).astype(np.bool)
    for i,v in enumerate(V):
        u = npr.uniform(0,1)
        thinned[i] = (u <= lam_t(v)/lam_ub(v))
        update_hp(lam_t,thinned,V)
    return thinned

def missing(eta,S):
    return npr.binomial(1,eta,size=len(S))

def generate(eta,ub_rate,hp_rate,T):
    V = processes.PoissonProcess('c',ub_rate(0)).sample_trajectory(T)
    T = thinning(V,ub_rate,hp_rate)
    S = V[np.where(T == 1)[0]] # PP with rate "rate_fxn"
    Z = missing(eta,S)
    print(Z)
    X = S[np.where(Z == 1)[0]]
    Y = S[np.where(Z == 0)[0]]
    return X,Y,V

def main():
    T = 5
    eta = 0.5
    def ub_rate(x): return 4
    def hp_rate(x): return np.sin(x) + 3

    X,Y,V = generate(eta,ub_rate,hp_rate,T)

    fig,ax = plt.subplots(1,2,figsize=(8,8))
    ax[0].plot(V,3*np.ones(len(V)),'kx')
    ax[0].plot(Y,2*np.ones(len(Y)),'bx')
    ax[0].plot(X,np.ones(len(X)),'rx')

    samples = run_gibbs(X,10,ub_rate,hp_rate,eta,missing)
    aggV = np.array(samples['V'])
    aggY = np.array(samples['Y'])

    xbarY = np.mean(aggY,axis=1)
    sstdY = np.std(aggY,axis=1)
    ax[1].errorbar(np.arange(len(Y)),xbarY,yerr=1.96*sstdY,alpha=0.5,fillstyle='full')
    ax[1].set_title("Summary of Y")
    
    plt.show()

def run_gibbs(X,T,ub_rate,hp_rate,eta,missing_fxn):
    nIters = 10**2
    Y0,V0 = None,None
    state = {'X':X, 'Y': Y0, 'V': V0,
             'ub_rate':ub_rate,
             'hp_rate':hp_rate,
             'eta':eta}
    agg_samples = {'Y':[],'V':[]}
    for i in range(nIters):
        
        # take gibbs step
        Y = gibbs_sample_Y(state)
        V = gibbs_sample_V(state)

        # book keeping
        sample = {'Y':Y, 'V': V}
        append_sample(sample,agg_samples)
        update_state(state,sample)

    return agg_samples
    
def gibbs_sample_Y(state):
    
    return Ynew

def append_sample(sample,agg_samples):
    for key,value in sample.items():
        agg_samples[key].append(value)

def update_state(state,sample):
    for key,value in sample.items():
        state[key] = value
    return state

if __name__ == "__main__":
    main()
    
