"""
---------------------------------
| Spencer Bertsch               |
| ENGG 177                      |
| Dartmouth College             |
| Spring 2022                   |
| Homework #2 - Question #3     |
---------------------------------

This script is designed to represent the options problem outlined in question #1 of homework #3 for ENGG 177. 

I will create an MDP similar to the "Random MDP" created as a template for this course, and hopefully 
it will yield realistic results when used in the provided backward induction algorithm. 
"""

# Loading packages
import numpy as np
import math

# Problem Parameters: 
rho: int = 1
c: int = 0

# Function to generate an arbitrary MDP
def Spine_MDP(n_T, n_S, n_A, r_min, r_max, seed=0):

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

            # TPM = np.random.rand(n_S, n_S)  # <-- random transition matrix 
            # here we can define a new transition matrix 
            TPM = np.array([[0.7, 0.3, 0, 0], [0, 0.5, 0.5, 0], [0, 0, 0.4, 0.6] ,[0, 0, 0, 1]], np.int8)
            TPM = TPM.T/TPM.sum(axis=1)  # <-- do we still need this? 
            
            # add the current time (t) and action (a) to the P matrix outside the loop 
            P[:, :, t, a] = TPM.T

            # generate the reward for being in state s_t and taking action (a) 
            for s in range(1, n_S, 1):
                print(f'STATE: {s+1}')

                # here the states will iterate over the array: [1, 2, 3, 4]

                # add the rewards for each action here (defer for one month or fix the spinal disk)
                if a == 0: 
                    r[s, t, a] = math.e ** (-rho * (s+1))
                else: 
                    r[s, t, a] = -c

    #Generate terminal rewards
    np.random.seed(seed)
    rterm = np.random.randint(r_min, r_max, size=n_S)

    return P, r, rterm