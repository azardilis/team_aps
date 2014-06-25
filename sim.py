import numpy as np
from functools import partial
from collections import namedtuple

AnabolicModel = namedtuple("AnabolicModel", ["S", "rate_funcs"])

def calc_current_rates(rate_funcs, state):
    return list(rf(state) for rf in rate_funcs)

def simulate_model(model, n_steps):
    SimResults = namedtuple('SimResults', ['trajectories', 'wait_times'])
    n_species = np.shape(model.S)[0]
    state_out  = np.zeros((n_steps+1, n_species))
    w_times = np.zeros(n_steps+1)
    state = np.zeros(n_species)
    state_out[0, :] = state
    t = 0.0
    w_times[0] = t
    
    for i in xrange(1, n_steps+1):
        curr_rates = calc_current_rates(model.rate_funcs, state)
        dt = np.random.exponential(1/np.sum(curr_rates))
        mres = np.random.multinomial(1, pvals=curr_rates/np.sum(curr_rates))
        ri = np.nonzero(mres)[0][0]
        state = state + model.S[:, ri]
        state_out[i, :] = state
        t += dt
        w_times[i] = dt

    return SimResults._make([state_out, w_times])

def get_rate_funcs(la, beta, C):
    # create rate functions for anabolic process
    # based on give params
    prod = partial(lambda x, la: la, la=la)
    deg1 = partial(lambda x, beta: beta*x[0], beta=beta)
    deg2 = partial(lambda x, beta: beta*x[1], beta=beta)
    complex_form = partial(lambda x, C: C*x[0]*x[1], C=C)

    return [prod, deg1, prod, deg2, complex_form]

def init_model(la, beta, C):
    # S : Stoichiometric matrix species x reactions
    S = np.array([[1, -1, 0, 0, -1],
                  [0, 0, 1, -1, -1]])
    rate_funcs = get_rate_funcs(la, beta, C)
    mod = AnabolicModel._make([S, rate_funcs])

    return mod

def get_vars(sim_res):
    n_species = np.shape(sim_res.trajectories)[1]
    traj = sim_res.trajectories[500:, :]
    w_times = sim_res.wait_times[500:]
    species_var = []
    
    for i in xrange(n_species):
        vals = traj[:, i]
        av = np.average(a=vals, weights=w_times)
        var = np.average(a=(vals-av)**2, weights=w_times)
        species_var.append(var/av**2)

    return species_var

def get_var_dists(mod, n_iter, n_steps):
    # Get the distribution of the variances of the species
    # for n_iter realisations of n_steps each
    n_species = np.shape(mod.S)[0]
    var_dists = np.zeros((n_iter, n_species))

    for i in xrange(n_iter):
        sim_res = simulate_model(mod, n_steps)
        vars = get_vars(sim_res)
        var_dists[i, :] = vars

    return var_dists

def get_var_devs(var_dists):
    # Returns var deviations of 2 species across simulation trajectories
    var_devs = list(np.abs(var_dists[i, 0]-var_dists[i, 1]) for i in range(np.shape(var_dists)[0]))

    return np.array(var_devs)

def plot_results(sim_res):
    # plot one example trace of the gillespie algorithm
    # for a particular set of params
    t = np.cumsum(sim_res)
    plt.plot(t, sim_res.trajectories)
    plt.show()
    # TODO: add some annotation to the graph!

def sim_anab_model():
    mod = init_model(la=0.5, beta=0.3, C=1.0)
    sim_results = simulate_model(model=mod, n_steps=100)

    return sim_results



    

    
    

    
    
    








