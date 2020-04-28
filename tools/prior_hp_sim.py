"""
This tools runs various Hawkes Process Simulations
"""

# python imports
import pprint,sys
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from functools import partial
from easydict import EasyDict as edict

# project imports
import _init_paths
import python_modules.processes as processes
import python_modules.base_plot as bplot
from hp.rate_fxn import HawkesProcessRate


def default_hp_rate():
    def base(x): return 1
    def alpha(k): return 1
    def beta(delta_t,k):
        tau = 50
        return np.exp(-delta_t*tau)
    hp_rate = HawkesProcessRate(base,alpha,beta)
    return hp_rate

def thinning_log_prob(V,L,ub_rate,hp_rate):
    hp_rate.reset()
    log_prob = 0
    Sp = []
    for event,l in zip(V,L):
        t,k = event,-1
        #t,k = event
        if l == 1: Sp.append(t)
        hp_rate.update((t,k))
        log_prob += np.log(hp_rate(t)) -  np.log(ub_rate(t))
    return log_prob

def thinning(V,ub_rate,hp_rate):
    # ub_rate :: the generating, upper-bound lambda rate
    # hp_rate :: the target lambda rate; e.g. the hawkes process rate
    thinned = np.zeros(len(V)).astype(np.bool)
    for i,event in enumerate(V):
        t,k = event,-1
        u = npr.uniform(0,1)
        thinned[i] = (u <= hp_rate(t)/ub_rate(t))
        if thinned[i] == 1: hp_rate.update((t,k))
    return thinned

def missing(eta,S):
    Zraw = npr.binomial(1,eta,size=len(S))
    return Zraw

def augmented_Z(Zraw,L):
    # put in -1's for thinned event
    # this is just to simplify _writing_ the code.
    # no consideration on performance.
    Z = -np.ones(len(L)).astype(np.int)
    print(np.sum(L == 1),len(Zraw))
    Z[np.where(L == 1)] = Zraw
    return Z
    
def explicit_state(Z,L,V):
    U = V[np.where(L == 0)[0]] # PP with rate "rate_fxn"
    S = V[np.where(L == 1)[0]] # PP with rate "rate_fxn"
    Z = Z[np.where(Z != -1)[0]] # just in case we have an 'augmented' Z
    X = S[np.where(Z == 1)[0]]
    Y = S[np.where(Z == 0)[0]]
    return X,Y,S,Z,U,L,V

def explicit_state_s(state):
    Z,L,V = state['Z'],state['L'],state['V']
    return explicit_state(Z,L,V)

def generate(eta,pp_rate,hp_rate,T):
    hp_rate.reset()
    V = processes.PoissonProcess('c',pp_rate(0)).sample_trajectory(T)
    L = thinning(V,pp_rate,hp_rate)
    U = V[np.where(L == 0)[0]] # PP with rate "rate_fxn"
    S = V[np.where(L == 1)[0]] # PP with rate "rate_fxn"
    Z = missing(eta,S)
    X = S[np.where(Z == 1)[0]]
    Y = S[np.where(Z == 0)[0]]
    return X,Y,Z,U,L,V

def example_prior_sample():
    T = 8
    eta = 0.5
    def pp_rate(x): return 6
    hp_rate = default_hp_rate()

    X,Y,Z,U,L,V = generate(eta,pp_rate,hp_rate,T)
    plot_sample(Z,L,V)

def plot_sample(Z,L,V):
    X,Y,S,Z,U,L,V = explicit_state(Z,L,V)
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    hV, = ax.plot(V,5*np.ones(len(V)),'kx',markersize=10)
    hU, = ax.plot(U,4*np.ones(len(U)),'cx',markersize=10)
    hS, = ax.plot(S,3*np.ones(len(S)),'bx',markersize=10)
    hY, = ax.plot(Y,2*np.ones(len(Y)),'gx',markersize=10)
    hX, = ax.plot(X,np.ones(len(X)),'rx',markersize=10)
    h = [hV,hU,hS,hY,hX]
    l = ['V','U','S','Y','X']
    bplot.add_legend(ax,"Names",l,h,fontsize=14)
    ax.set_title("Generative Model of Censored HP",size=15)
    plt.show()

def sample_prior(N=100):
    print(f"Sampling Prior N = {N}")

    T = 8
    eta = 0.5
    def pp_rate(x): return 6
    hp_rate = default_hp_rate()

    agg = {'V':[],'L':[],'Z':[]}
    #agg = {'X':[],'Y':[],'Z':[],'U':[],'L':[],'V':[]}
    for i in range(N):
        X,Y,Z,U,L,V = generate(eta,pp_rate,hp_rate,T)
        sample = {'V':V,'L':L,'Z':Z}
        append_samples(agg,sample)
    return agg

def gibbs_sampler(state,N=100):
    """
    Collecting samples of V,L,Z
    V :: all samples from a dominating poisson process
    L :: the binary mask for the thinning procedure that split V into U and S
    Z :: the binary mask for censoring events into sets X and Y
    Remark: Z is _actually_ THREE values (-1,0,1) to simplify writing code.
    """

    # setup a bunch of parameters
    T = 8
    eta = 0.5
    def pp_rate(x): return 6
    hp_rate = default_hp_rate()
    params = edict({'hp_rate':hp_rate,'pp_rate':pp_rate,'eta':eta,'rate_ub':pp_rate(0)})

    # collect info
    agg = {'V':[],'L':[],'Z':[]}

    # run sampler
    for i in range(N):

        # resample thinned events
        sample_U(state,params,T)
        
        # switch events between "Y" and "U"
        mwg_resample_l(state,params) # updates in place

        print(f"Gibbs sample {i}/{N}")

        append_samples(agg,state)
    return agg

def sample_U(state,params,T):
    def u_rate_skel(rate_fxn,x): return params.pp_rate(x) - rate_fxn(x)
    hp_rate = params.hp_rate
    hp_rate.reset()
    S = getS(state)
    hp_rate.set_hist([(s,-1) for s in S])
    u_rate = partial(u_rate_skel,hp_rate)
    U = processes.PoissonProcess('n',u_rate,rate_ub=params.rate_ub).sample_trajectory(T)
    update_state_with_U(state,U)
    
def update_state_with_U(state,U):
    """
    Given the new thinned events, we recreate V
    and change the number of zeros in L to match len(U)
    """
    S = get_S_from_state(state)
    Vnew = np.r_[S, U]
    I = np.argsort(Vnew) # this is the reordering to preserve X locations
    Vnew = Vnew[I]
    state['V'] = Vnew

    Lnew_s = np.ones(len(S))
    Lnew_u = np.zeros(len(U))
    Lnew = np.r_[Lnew_s,Lnew_u].astype(np.int)
    Lnew = Lnew[I]
    state['L'] = Lnew

    Z = state['Z']
    Zraw = Z[np.where(Z != -1)]
    Znew = np.r_[Zraw,-np.ones(len(U))]
    Znew = Znew[I].astype(np.int)
    state['Z'] = Znew

def getS(state):
    return get_S_from_state(state)

def get_S_from_state(state):
    V = state['V']
    L = state['L']
    U = V[np.where(L == 1)]
    return U

def mwg_resample_l(state,params):
    """
    state ::  dict
          'V' : a list of tuples with (time,mark)
          'L' : a list of 0,1 indicated thinned or hawkes events
          'Z' : a list of -1,0,1 indicating thinned, censored, or observed
    Note augmented values for Z in the state are to reduce computation.
    While we could run the algorithm with only updating Z, we include L for clarity.
    
    params :: easydict
         'hp_rate'
         'pp_rate'
         'eta'
    """
    
    V,Z = state['V'],state['Z']

    for event_index,event_type,event in zip(range(len(V)),Z,V):
        if event_type == 1: continue # observed
        elif event_type == 0: # event in Y; move to U?
            alpha_y2u = compute_log_alpha_y2u(event,event_index,state,params)
            u = np.log(npr.uniform(0,1))
            if u <= alpha_y2u: #accept
                state['L'][event_index] = 0
                state['Z'][event_index] = -1
        elif event_type == -1: # "else" ... event in U; move to Y?
            alpha_u2y = compute_log_alpha_u2y(event,event_index,state,params)
            u = np.log(npr.uniform(0,1))
            if u <= alpha_u2y: #accept
                state['L'][event_index] = 1
                state['Z'][event_index] = 0


def compute_log_alpha_y2u(event,event_index,state,params):
    V,L,Z = state['V'],state['L'],state['Z']
    log_pVLZ = compute_state_ll(V,L,Z,params)

    assert L[event_index] == 1, "L[event_index] 0: y2u"

    L_n,Z_n = np.copy(L),np.copy(Z)
    L_n[event_index] = 0
    Z_n[event_index] = -1
    log_pVLZ_n = compute_state_ll(V,L_n,Z_n,params)

    lalpha = log_pVLZ_n - log_pVLZ

    return lalpha

def compute_log_alpha_u2y(event,event_index,state,params):
    V,L,Z = state['V'],state['L'],state['Z']
    log_pVLZ = compute_state_ll(V,L,Z,params)

    assert L[event_index] == 0, "L[event_index] 1: u2y"

    L_n = np.copy(L)
    L_n[event_index] = 1
    Z_n = np.copy(Z)
    Z_n[event_index] = 0
    log_pVLZ_n = compute_state_ll(V,L_n,Z_n,params)

    lalpha = log_pVLZ_n - log_pVLZ

    return lalpha
    
def compute_state_ll(V,L,Z,params):
    """
    params. ...
    Need: pp_rate,hp_rate,eta
    """
    
    # V; we don't need this it cancels on both. V doesn't change
    # def pp_rate(x): return 6
    # V_ll_fxn = poisson.PoissonProcess('c',pp_rate(0)).ll
    # V_ll = V_ll_fxn(V)

    # L
    L_ll = thinning_log_prob(V,L,params.pp_rate,params.hp_rate)
    
    # Z
    Z_ll = easy_z_ll(Z,params.eta)
    
    return L_ll + Z_ll

def easy_z_ll(Z,eta):
    a = np.sum(Z == 1) * np.log(eta)
    b = np.sum(Z == 0) * np.log(1 - eta)
    return a + b

def append_samples(agg,sample):
    for key,val in sample.items():
        agg[key].append(val)

def get_init_sample():
    sample = sample_prior(N=1)
    state = {}
    for key in ['V','L','Z']: state[key] = sample[key][0]
    state['Z'] = augmented_Z(state['Z'],state['L'])
    for key in state.keys(): print(len(state[key]),key)
    return state

def main():
    # example_prior_sample()
    # psamples = sample_prior()

    state_init = get_init_sample()
    gsamples = gibbs_sampler(state_init,N=100)

    Zall,Lall,Vall = gsamples['Z'],gsamples['L'],gsamples['V']
    plot_sample(Zall[0],Lall[0],Vall[0])


if __name__ == "__main__":
    main()
    
