import numpy as np
from functools import partial
from collections import namedtuple
import matplotlib.pyplot as plt

AnabolicModel = namedtuple("AnabolicModel", ["S", "rate_funcs"])

def calc_current_rates(rate_funcs, state):
    return list(rf(state) for rf in rate_funcs)
 
def simulate_model(model, n_steps):
    # Given a model simulates it using the Gillespie algorithm
    # for n_steps number of steps
    #
    # args:
    #        model: tuple containing the stoichiometric matrix of the 
    #               reactions(S) and rate functions(rate_funcs)
    # Returns:
    #        tuple with species trajectories(trajectories) and wait times
    SimResults = namedtuple('SimResults', ['trajectories', 'wait_times'])
    n_species = np.shape(model.S)[0]
    state_out  = np.zeros((n_steps+1, n_species))
    w_times = np.zeros(n_steps+1)
    state = np.zeros(n_species)
    state_out[0, :] = state
    w_times[0] = 0.0
    
    for i in xrange(1, n_steps+1):
        curr_rates = calc_current_rates(model.rate_funcs, state)
        dt = np.random.exponential(1/np.sum(curr_rates))
        mres = np.random.multinomial(1, pvals=curr_rates/np.sum(curr_rates))
        ri = np.nonzero(mres)[0][0]
        state = state + model.S[:, ri]
        state_out[i, :] = state
        w_times[i] = dt

    return SimResults._make([state_out, w_times])

def get_rate_funcs(la, beta, C):
    # Creates rate functions for anabolic process
    # based on given parameters
    prod = partial(lambda x, la: la, la=la)
    deg1 = partial(lambda x, beta: beta*x[0], beta=beta)
    deg2 = partial(lambda x, beta: beta*x[1], beta=beta)
    complex_form = partial(lambda x, C: C*x[0]*x[1], C=C)
    
    return [prod, deg1, prod, deg2, complex_form]

def init_model(la, beta, C):
    # Initialises anabolic process model with the 
    # stoichiometry matrix and rate functions
    #
    # S : Stoichiometric matrix species x reactions
    S = np.array([[1, -1, 0, 0, -1],
                  [0, 0, 1, -1, -1]])
    rate_funcs = get_rate_funcs(la, beta, C)
    mod = AnabolicModel._make([S, rate_funcs])

    return mod

def get_res_dist(sim_res):
    # Given a specific realisation of the process
    # returns the averages and variance of the species in
    # two tuples.
    n_species = np.shape(sim_res.trajectories)[1]
    traj = sim_res.trajectories[5000:, :]
    w_times = sim_res.wait_times[5000:]
    species_var = []
    species_avg = []
    
    for i in xrange(n_species):
        vals = traj[:, i]
        av = np.average(a=vals, weights=w_times)
        var = np.average(a=(vals-av)**2, weights=w_times)
        species_var.append(var/av**2)
        species_avg.append(av)

    return species_var, species_avg

def get_devs(dists):
    # Returns deviations of statistics between 2 species
    # across different simulations
    devs = list(dists[i, 0]-dists[i, 1] for i in range(np.shape(dists)[0]))

    return np.array(devs)

def calc_approx_var(avgs, params):
    beta = params[1]
    C = params[2]
    x1 = avgs[0]
    x2 = avgs[1]
    
    E = (C*x1*x2) / (C*x1*x2 + beta*x1)
    approx_var = (1/x1) * ((1-(E**2/2)) / (1-E**2))
    
    return approx_var
    
def get_approx_vars(avgs, params):
    # avgs: species averages for different simulations
    # params: parameters used for each of the simulations
    #
    # Returns:
    # avars: list of approximate CVs for the simulations
    n_sims = np.shape(avgs)[0]
    avars = []
    for i in xrange(n_sims):
        avar = calc_approx_var(avgs[i, :], params[i])
        avars.append(avar)
        
    return avars
    
def calc_neg_flux(avgs, params):
    beta = params[1]
    C = params[2]
    
    x1 = avgs[0]
    x2 = avgs[1]
    
    return C*x1*x2 + beta*x1
    
def get_neg_fluxes(avgs, params):
    # Given averages and parameters for different simulation
    # returns the average negative fluxes
    n_sims = np.shape(avgs)[0]
    nfs = []
    
    for i in xrange(n_sims):
        nf = calc_neg_flux(avgs[i, :], params[i])        
        nfs.append(nf)
        
    return nfs
    
def plot_results(sim_res):
    # Plots one example trace for one simulation
    t = np.cumsum(sim_res.wait_times)
    plt.plot(t, sim_res.trajectories[:, 0], label=r"$x_1$")
    plt.plot(t, sim_res.trajectories[:, 1], label=r"$x_2$")
    plt.xlabel("t")
    plt.ylabel("# of molecules")
    plt.legend()
    plt.show()

def sim_anab_model():
    mod = init_model(la=0.5, beta=0.3, C=1.0)
    sim_res = simulate_model(model=mod, n_steps=100)
    vars, avgs = get_res_dist(sim_res)
    print vars, avgs
