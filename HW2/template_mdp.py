# *********************
# Random MDP Genration
# *********************
# Loading packages
import numpy as np

# Function to generate an arbitrary MDP
def random_MDP(n_T, n_S, n_A, r_min, r_max, seed=0):

    #This function generates random parameters for a finite horizon MDP

    """"
    Inputs:
        n_T = number of stages
        n_S = number of states
        n_A = number of actions
        r_min = minimum reward
        r_max= maximum reward
    """""

    #Generate transition probabilities and rewards
    P = np.empty((n_S, n_S, n_T, n_A)); P[:] = np.nan
    r = np.empty((n_S, n_T, n_A)); r[:] = np.nan

    for t in range(n_T):
        for a in range(n_A):
            np.random.seed(seed); 
            seed += 1  # seeds for pseudo-random number generator
            TPM = np.random.rand(n_S, n_S)
            TPM = TPM.T/TPM.sum(axis=1)
            P[:, :, t, a] = TPM.T
            for s in range(n_S):
                r[s, t, a] = np.random.randint(r_min, r_max)

    #Generate terminal rewards
    np.random.seed(seed)
    rterm = np.random.randint(r_min, r_max, size=n_S)

    return P, r, rterm