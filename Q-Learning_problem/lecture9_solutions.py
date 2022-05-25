# *******************************************
# Solutions to lecture 9's in-class exercise
# *******************************************

# Loading packages
import numpy as np
from template_value_iteration import value_iteration
from template_policy_iteration import policy_iteration
from template_modified_policy_iteration import mod_policy_iteration

# --------------
# Parameters
# --------------

# Reward vectors
## Order of decision rules: send to both classes, send to low class, send to high class, do not send
r = np.array([[5, 5, 10, 10], [35, 25, 35, 25]]) # reward vectors

# Transition probability matrices
## Order of decision rules: send to both classes, send to low class, send to high class, do not send
PB = np.array([[0.3, 0.7], [0.2, 0.8]]); PL = np.array([[0.3, 0.7], [0.6, 0.4]])
PH = np.array([[0.5, 0.5], [0.2, 0.8]]); PN = np.array([[0.5, 0.5], [0.6, 0.4]])
P = np.dstack((PB, PL, PH, PN))

# Algorithmic parameters
discount = 0.9 # discount factor
epsilon = 0.1 # suboptimality tolerance
N = 200 # maximum number of iterations for value iteration
m = int(1e02) # number of iterations for partial policy evaluation in modified policy iteration
v_init = np.repeat(0, P.shape[0]) # initial value functions (for value iteration and modified policy iteration)
d_init = 0 # initial decision rule (for policy iteration)

# --------------
# Solving MDP
# --------------

Q_vi, v_vi, d_vi = value_iteration(P, r, v_init, discount, epsilon, N) # obtaining epsilon-optimal policies using value iteration
Q_pi, v_pi, d_pi = policy_iteration(P, r, d_init, discount) # obtaining optimal policies using policy iteration
Q_mpi, v_mpi, d_mpi = mod_policy_iteration(P, r, v_init, discount, epsilon, m) # obtaining epsilon-optimal policies using modify policy iteration
