# *****************************
# Backward Induction Algorithm
# *****************************

# Loading packages
import numpy as np
from template_mdp import random_MDP

# Backward induction algorithm function
def backward_induction(P, r, rterm, discount):

    """
    Inputs:
    P: S x S x T x A array of transition probabilities
    r: S x T x A array of rewards
    rterm: S array of terminal rewards 
    discount: discount factor between 0 and 1
    """""

    """
    Outputs:
        Q: action-value functions
        v: optimal value functions
        pi: optimal policy
    """""

    # Extracting parameters
    S = P.shape[0]  # number of states
    T = P.shape[2]  # number of stages (excluding terminal stage)
    A = P.shape[3]  # number of actions

    # Storing MDP calculations
    Q = np.full((S, T, A), np.nan) # stores action-value functions
    v = np.full((S, T), np.nan) # stores optimal value function values
    pi = np.full((S, T), np.nan) # stores optimal policy

    for t in reversed(range(T)): # number of decision epochs remaining
        for s in range(S): # each state
            for a in range(A): # each action

                # Computing action-value functions
                if t == max(range(T)): # one decision remaining to be made in planning horizon
                    Q[s, t, a] = r[s, t, a] + discount*np.sum([P[s, ss, t, a]*rterm[ss] for ss in range(S)])  # terminal condition
                else:
                    Q[s, t, a] = r[s, t, a] + discount*np.sum([P[s, ss, t, a]*rterm[ss] for ss in range(S)])  # backward induction

            # Optimal value function and policies
            v[s, t] = np.amax(Q[s, t, :])  # optimal value function at state s and stage t
            pi[s, t] = np.argmax(Q[s, t, :])  # optimal decision rule

    return Q, v, pi

# Example
## Generating random MDP patameters
n_T = 3; n_S = 4; n_A = 5; r_min = 100; r_max = 200; seed = 15
P, r, rterm = random_MDP(n_T, n_S, n_A, r_min, r_max, seed)

## Calculating value functions and policies using backward induction
Q, v, pi = backward_induction(P, r, rterm, discount=0.97)
