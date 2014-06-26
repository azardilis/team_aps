import numpy as np
from itertools import product
import sim
import matplotlib.pyplot as plt
import operator

def calc_dists(L, B, C, n_steps):
    # Given ranges for params calc variances, averages
    # for each parameter combination
    vars = []
    avgs = []
    
    for la, beta, c in product(L, B, C):
        mod = sim.init_model(la, beta, c)
        sim_res = sim.simulate_model(mod, n_steps)
        v, a = sim.get_res_dist(sim_res)
        vars.append(v)
        avgs.append(a)

    return np.array(vars), np.array(avgs)


def plot_vars(vars):
    n_points = 20
    C = np.linspace(1.0, 3.0, n_points)
    L = np.linspace(3.0, 5.0 ,n_points)

    vars_x1 = np.reshape(vars[:, 0], (n_points, n_points))
    vars_x2 = np.reshape(vars[:, 1], (n_points, n_points))

    plt.subplot(121)
    plt.imshow(vars_x1, interpolation='none', aspect=20/20.)
    plt.xlabel(r'C')
    plt.ylabel(r'$\lambda$')
    plt.xticks(np.arange(0, n_points, 4), np.around(C[np.arange(0, n_points, 4)], 1))
    plt.yticks(np.arange(0, n_points, 4), np.around(L[np.arange(0, n_points, 4)], 1))
    plt.title(r'CV $x_2$')
    cticks1 = np.around(np.linspace(np.min(vars_x1), np.max(vars_x1), 4), 1)
    plt.colorbar(shrink=.5, aspect=10., ticks=cticks1)
    
    plt.subplot(122)
    plt.imshow(vars_x2, interpolation='none', aspect=20/20.)
    plt.xlabel(r'C')
    plt.ylabel(r'$\lambda$')
    plt.xticks(np.arange(0, n_points, 4), np.around(C[np.arange(0, n_points, 4)], 1))
    plt.yticks(np.arange(0, n_points, 4), np.around(L[np.arange(0, n_points, 4)], 1))
    plt.title(r'CV $x_2$')
    cticks2 = np.around(np.linspace(np.min(vars_x2), np.max(vars_x2), 4), 1)
    plt.colorbar(pad=0.05, shrink=.5, aspect=10., ticks=cticks2)

    
    plt.show()

def plot_dev_hist(dists):
    # Given pairs of either species vars or averages plot
    # a histogram of their deviations
    devs = sim.get_devs(dists)
    plt.hist(devs, alpha=0.8)
    plt.title("Deviations of the CVs between species")
    plt.xlabel(r"$|(\sigma_{x_1}^2/\langle x_1 \rangle^2)-(\sigma_{x_2}^2/\langle x_2 \rangle^2)|$")
    plt.ylabel("Frequency")
    plt.show()
    
def plot_approx_vars(avars, vars):
    # plot of the exact variances against
    # the approximate variances calculate in terms
    # of avgs and Efficiencies
    plt.plot(vars, avars, 'o', color='w')
    plt.xlabel("Exact variance")
    plt.ylabel("Approximate variance")
    plt.show()
    plt.close()
    plt.hist(np.abs(vars-avars), alpha=0.8)
    plt.xlabel("|Exact var - approx var|")
    plt.ylabel("Frequency")
    plt.show()
    



        
