"""
---------------------------------
| Spencer Bertsch               |
| ENGG 177                      |
| Dartmouth College             |
| Spring 2022                   |
| Homework #2 - Question #1     |
---------------------------------

This script is designed to represent the options problem outlined in question #1 of homework #2 for ENGG 177. 

I will create an MDP similar to the "Random MDP" created as a template for this course, and hopefully 
it will yield realistic results when used in the provided backward induction algorithm. 
"""

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

    # Initialize empty matrices for transition probabilities
    P = np.empty((n_S, n_S, n_T, n_A))
    P[:] = np.nan
    
     # Initialize empty matrices for rewards
    r = np.empty((n_S, n_T, n_A))
    r[:] = np.nan

    # we step forward once for each epoch before we get to T
    for t in range(n_T):
        # we take all the actions possible from the current state s_t
        for a in range(n_A):

            # if action == 0 then we exercise option, if action == 1 we continue

            # seeds for pseudo-random number generator
            np.random.seed(seed)
            seed += 1  

            TPM = np.random.rand(n_S, n_S)
            TPM = TPM.T/TPM.sum(axis=1)
            
            # add the current time (t) and action (a) to the P matrix outside the loop 
            P[:, :, t, a] = TPM.T

            # generate the reward for being in state s_t and taking action (a) 
            for s in range(n_S):
                # how to we add the correct rewards here?? 
                if a == 0: 
                    r[s, t, a] = s
                r[s, t, a] = np.random.randint(r_min, r_max)

    #Generate terminal rewards
    np.random.seed(seed)
    rterm = np.random.randint(r_min, r_max, size=n_S)

    return P, r, rterm