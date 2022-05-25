# ****************************
# Temporal Difference Methods
# ****************************

# Loading packages
import numpy as np
from template_policy_iteration import policy_iteration # included for comparison purposes

# ------------------
# Policy evaluation
# ------------------

# TD(0) algorithm (assuming 1/n step-size)
def td_0(pi, P, r, s_term, discount, N):
    """
    Inputs:
        pi: S x A array of probabilities of selecting each action at every state
        P: S x S x A array of transition probabilities (likely unknown in practice)
        r: S x A array of rewards
        s_term: index of terminal state (used to end episodes)
        discount: discount factor between 0 and 1
        N: maximum number of episodes
    """""

    """
    Outputs:
        v_hat: estimate of value functions associated with the policy of interest
    """""

    # Extracting parameters
    S = P.shape[0]  # number of states

    # Initializing parameters
    v_hat = np.zeros(S)  # initializing value functions
    N_s = np.zeros(S)  # matrix to store number of observations in each state (for step-size)
    seed = 100  # initial seed for pseudo-random number generator (for reproducibility)

    for n in range(N):  # each episode
        np.random.seed(seed); seed += 1  # establishing seed of pseudo-random numbers
        s_now = np.random.choice(np.arange(S)) # generating initial state
        stop = 0  # indicator of when we have reached a terminal state

        while stop != 1:  # continue in episode until we reach end of episode

            # Checking if we are in a terminal state
            if s_now == s_term:  # terminal state
                stop = 1  # indicator of when we have reached a terminal state
            else: # nonterminal state
                # Selecting action according to policy of interest
                np.random.seed(seed); seed += 1  # establishing seed
                a = np.random.choice(np.arange(pi.shape[1]), p=pi[s_now, :])  # selecting next action

                # Determining next state from transition probabilities
                ## (if transition probabilities are not known in practice the next state must be observed)
                np.random.seed(seed); seed += 1  # establishing seed
                s_next = np.random.choice(np.arange(S), p=P[s_now, :, a])  # sampling next state

                # Updating estimate of value function
                N_s[s_now]+=1; alpha = 1/N_s[s_now]  # establishing step-size parameter (using the Harmonic series as the step-size)
                v_hat[s_now] += alpha*((r[s_now, a]+discount*v_hat[s_next]) - v_hat[s_now]) # updating estimate

                # Updating current state
                s_now = s_next

    return v_hat

# ---------------------------------------
# Finding approximately optimal policies
# ---------------------------------------

# Sarsa algorithm (assuming 1/n step-size)
def sarsa(P, r, s_term, discount, N, epsilon=1):
    """
    Inputs:
        P: S x S x A array of transition probabilities (likely unknown in practice)
        r: S x A array of rewards
        s_term = index of terminal state (used to end episodes)
        discount: discount factor between 0 and 1
        N: maximum number of episodes
        epsilon: initial value of exploration parameter (it decreases over the episodes)
    """""

    """
    Outputs:
        Q_hat: estimate of action-value functions
        pi: S x A array of approximately optimal probabilities of selecting each action at every state
    """""

    # Extracting parameters
    S = P.shape[0]  # number of states
    A = P.shape[2]  # number of actions

    # Initializing parameters
    Q_hat = np.zeros((S, A)) # initializing action-value functions
    pi = np.ones((S, A))*epsilon/A # assigning epsilon/A probability of selection to all actions
    greedy = np.argmax(Q_hat, axis=1) # identifying greedy action in each state
    pi[np.arange(pi.shape[0]), greedy] += (1-epsilon) # increasing the probability of selection of the best action in each state
    N_sa = np.zeros((S, A))  # matrix to store number of observations in each state and action pair (for step-size)
    seed = 100 # initial seed for pseudo-random number generator

    for n in range(N): # each episode

        # Generating initial state
        np.random.seed(seed); seed += 1  # establishing seed of pseudo-random numbers
        s_now = np.random.choice(np.arange(S)) # generating initial state
        stop = 0  # indicator of when we have reached a terminal state

        # Selecting current action according to policy of interest
        np.random.seed(seed); seed += 1  # establishing seed
        a_now = np.random.choice(np.arange(pi.shape[1]), p=pi[s_now, :])  # selecting next action

        while stop != 1:  # continue in episode until we reach end of episode

            # Checking if we are in a terminal state
            if s_now == s_term:  # terminal state
                stop = 1  # indicator of when we have reached a terminal state
            else: # nonterminal state
                # Determining next state from transition probabilities
                ## (if transition probabilities are not known in practice the next state must be observed)
                np.random.seed(seed); seed += 1  # establishing seed
                s_next = np.random.choice(np.arange(S), p=P[s_now, :, a_now]) # sampling next state

                # Selecting next action according to policy of interest
                np.random.seed(seed); seed += 1  # establishing seed
                a_next = np.random.choice(np.arange(pi.shape[1]), p=pi[s_next, :]) # selecting next action

                # Updating estimate of action-value function
                N_sa[s_now, a_now]+=1; alpha = 1/N_sa[s_now, a_now] # establishing step-size parameter (using the Harmonic series as the step-size)
                Q_hat[s_now, a_now] += alpha*((r[s_now, a_now]+discount*Q_hat[s_next, a_next]) - Q_hat[s_now, a_now]) # updating action-value function

                # Updating epsilon-greedy policy
                epsilon = 1/((n+1)+1)  # updating value of epsilon
                pi[s_now, :] = np.ones(A)*epsilon/A # assigning epsilon/A probability of selection to all actions
                greedy = np.argmax(Q_hat[s_now, :]) # identifying greedy action in current state
                pi[s_now, greedy] += (1-epsilon) # increasing the probability of selection of the best action in current state

                # Updating current state and action
                s_now = s_next; a_now = a_next

    return Q_hat, pi

# Q-learning algorithm (assuming 1/n step-size and epsilon-greed behavior policy)
def q_learning(P, r, s_term, discount, N, epsilon):
    """
    Inputs:
        P: S x S x A array of transition probabilities (likely unknown in practice)
        r: S x A array of rewards
        s_term = index of terminal state (used to end episodes)
        discount: discount factor between 0 and 1
        N: maximum number of episodes
        epsilon: value of exploration parameter
    """""

    """
    Outputs:
        Q_hat: estimate of action-value functions
        pi: S x A array of approximately optimal probabilities of selecting each action at every state
    """""

    # Extracting parameters
    S = P.shape[0]  # number of states
    A = P.shape[2]  # number of actions

    # Initializing parameters
    Q_hat = np.zeros((S, A)) # initializing action-value functions
    b = np.ones((S, A))*epsilon/A # assigning epsilon/A probability of selection to all actions
    greedy = np.argmax(Q_hat, axis=1) # identifying greedy action in each state
    b[np.arange(b.shape[0]), greedy] += (1-epsilon) # increasing the probability of selection of the best action in each state
    N_sa = np.zeros((S, A))  # matrix to store number of observations in each state and action pair (for step-size)
    seed = 100 # initial seed for pseudo-random number generator

    for n in range(N): # each episode

        # Generating initial state
        np.random.seed(seed); seed += 1  # establishing seed of pseudo-random numbers
        s_now = np.random.choice(np.arange(S)) # generating initial state
        stop = 0  # indicator of when we have reached a terminal state

        while stop != 1:  # continue in episode until we reach end of episode

            # Checking if we are in a terminal state
            if s_now == s_term:  # terminal state
                stop = 1  # indicator of when we have reached a terminal state
            else: # nonterminal state
                # Selecting current action according to behavior policy
                np.random.seed(seed); seed += 1  # establishing seed
                a_now = np.random.choice(np.arange(b.shape[1]), p=b[s_now, :])  # selecting next action

                # Determining next state from transition probabilities
                ## (if transition probabilities are not known in practice the next state must be observed)
                np.random.seed(seed); seed += 1  # establishing seed
                s_next = np.random.choice(np.arange(S), p=P[s_now, :, a_now]) # sampling next state

                # Selecting next action according to greedy policy
                np.random.seed(seed); seed += 1 # establishing seed
                a_next = np.argmax(Q_hat[s_next, :]) # selecting next action

                # Updating estimate of action-value function
                N_sa[s_now, a_now]+=1; alpha = 1/N_sa[s_now, a_now] # establishing step-size parameter (using the Harmonic series as the step-size)
                Q_hat[s_now, a_now] += alpha*((r[s_now, a_now]+discount*Q_hat[s_next, a_next]) - Q_hat[s_now, a_now]) # updating action-value function

                # Updating epsilon-greedy policy
                b[s_now, :] = np.ones(A)*epsilon/A # assigning epsilon/A probability of selection to all actions
                greedy = np.argmax(Q_hat[s_now, :]) # identifying greedy action in current state
                b[s_now, greedy] += (1-epsilon) # increasing the probability of selection of the best action in current state

                # Updating current state
                s_now = s_next

    # Identifying approximately optimal policy
    pi = np.argmax(Q_hat, axis=1)

    return Q_hat, pi

# Expected Sarsa algorithm (assuming 1/n step-size)
def expected_sarsa(P, r, s_term, discount, N, epsilon=1):
    """
    Inputs:
        P: S x S x A array of transition probabilities (likely unknown in practice)
        r: S x A array of rewards
        s_term = index of terminal state (used to end episodes)
        discount: discount factor between 0 and 1
        N: maximum number of episodes
        epsilon: initial value of exploration parameter (it decreases over the episodes)
    """""

    """
    Outputs:
        Q_hat: estimate of action-value functions
        pi: S x A array of approximately optimal probabilities of selecting each action at every state
    """""

    # Extracting parameters
    S = P.shape[0]  # number of states
    A = P.shape[2]  # number of actions

    # Initializing parameters
    Q_hat = np.zeros((S, A)) # initializing action-value functions
    pi = np.ones((S, A))*epsilon/A # assigning epsilon/A probability of selection to all actions
    greedy = np.argmax(Q_hat, axis=1) # identifying greedy action in each state
    pi[np.arange(pi.shape[0]), greedy] += (1-epsilon) # increasing the probability of selection of the best action in each state
    N_sa = np.zeros((S, A))  # matrix to store number of observations in each state and action pair (for step-size)
    seed = 100 # initial seed for pseudo-random number generator

    for n in range(N): # each episode

        # Generating initial state
        np.random.seed(seed); seed += 1  # establishing seed of pseudo-random numbers
        s_now = np.random.choice(np.arange(S)) # generating initial state
        stop = 0  # indicator of when we have reached a terminal state

        while stop != 1:  # continue in episode until we reach end of episode

            # Checking if we are in a terminal state
            if s_now == s_term:  # terminal state
                stop = 1  # indicator of when we have reached a terminal state
            else: # nonterminal state
                # Selecting current action according to policy of interest
                np.random.seed(seed); seed += 1  # establishing seed
                a_now = np.random.choice(np.arange(pi.shape[1]), p=pi[s_now, :])  # selecting next action

                # Determining next state from transition probabilities
                ## (if transition probabilities are not known in practice the next state must be observed)
                np.random.seed(seed); seed += 1  # establishing seed
                s_next = np.random.choice(np.arange(S), p=P[s_now, :, a_now]) # sampling next state

                # Updating estimate of action-value function
                N_sa[s_now, a_now]+=1; alpha = 1/N_sa[s_now, a_now] # establishing step-size parameter (using the Harmonic series as the step-size)
                Q_hat[s_now, a_now] += alpha*((r[s_now, a_now]+discount*np.sum([pi[s_next, a]*Q_hat[s_next, a] for a in range(A)]))
                                              - Q_hat[s_now, a_now]) # updating action-value function

                # Updating epsilon-greedy policy
                epsilon = 1/((n+1)+1)  # updating value of epsilon
                pi[s_now, :] = np.ones(A)*epsilon/A # assigning epsilon/A probability of selection to all actions
                greedy = np.argmax(Q_hat[s_now, :]) # identifying greedy action in current state
                pi[s_now, greedy] += (1-epsilon) # increasing the probability of selection of the best action in current state

                # Updating current state
                s_now = s_next

    return Q_hat, pi

# Double Q-learning algorithm (assuming 1/n step-size and epsilon-greedy behavior policy)
def double_q_learning(P, r, s_term, discount, N, epsilon):
    """
    Inputs:
        P: S x S x A array of transition probabilities (likely unknown in practice)
        r: S x A array of rewards
        s_term = index of terminal state (used to end episodes)
        discount: discount factor between 0 and 1
        N: maximum number of episodes
        epsilon: value of exploration parameter
    """""

    """
    Outputs:
        Q_hat: estimate of action-value functions
        pi: S x A array of approximately optimal probabilities of selecting each action at every state
    """""

    # Extracting parameters
    S = P.shape[0]  # number of states
    A = P.shape[2]  # number of actions

    # Initializing parameters
    Q_hat1 = np.zeros((S, A)); Q_hat2 = np.zeros((S, A)) # initializing action-value functions
    N1 = 0 # initializing counter for number of times Q_hat1 is updated (to account for the possibility of different number of samples in Q_hat1 and Q_hat2)
    b = np.ones((S, A))*epsilon/A # assigning epsilon/A probability of selection to all actions
    greedy = np.argmax(Q_hat1+Q_hat2, axis=1) # identifying greedy action in each state
    b[np.arange(b.shape[0]), greedy] += (1-epsilon) # increasing the probability of selection of the best action in each state
    N_sa1 = np.zeros((S, A)); N_sa2 = np.zeros((S, A))  # matrices to store number of observations in each state and action pair (for step-size)
    seed = 100 # initial seed for pseudo-random number generator

    for n in range(N): # each episode

        # Generating initial state
        np.random.seed(seed); seed += 1  # establishing seed of pseudo-random numbers
        s_now = np.random.choice(np.arange(S)) # generating initial state
        stop = 0  # indicator of when we have reached a terminal state

        while stop != 1:  # continue in episode until we reach end of episode

            # Checking if we are in a terminal state
            if s_now == s_term:  # terminal state
                stop = 1  # indicator of when we have reached a terminal state
            else: # nonterminal state
                # Selecting current action according to behavior policy
                np.random.seed(seed); seed += 1  # establishing seed
                a_now = np.random.choice(np.arange(b.shape[1]), p=b[s_now, :])  # selecting next action

                # Determining next state from transition probabilities
                ## (if transition probabilities are not known in practice the next state must be observed)
                np.random.seed(seed); seed += 1  # establishing seed
                s_next = np.random.choice(np.arange(S), p=P[s_now, :, a_now]) # sampling next state

                # Updating action-value function
                np.random.seed(seed); seed += 1 # establishing seed for next action selection
                u = np.random.uniform() # generating uniform random number
                if u < 0.5: # updating first estimate of action-value function
                    # Selecting next action according to greedy policy
                    a_next = np.argmax(Q_hat1[s_next, :])

                    # Updating first estimate of action-value function
                    N_sa1[s_now, a_now] += 1; alpha = 1/N_sa1[s_now, a_now]  # establishing step-size parameter (using the Harmonic series as the step-size)
                    Q_hat1[s_now, a_now] += alpha*((r[s_now, a_now]+discount*Q_hat2[s_next, a_next]) - Q_hat1[s_now, a_now]) # updating action-value function
                else: # updating second estimate of action-value function
                    # Selecting next action according to greedy policy
                    a_next = np.argmax(Q_hat2[s_next, :])

                    # Updating second estimate of action-value function
                    N_sa2[s_now, a_now] += 1; alpha = 1/N_sa2[s_now, a_now]  # establishing step-size parameter (using the Harmonic series as the step-size)
                    Q_hat2[s_now, a_now] += alpha*((r[s_now, a_now]+discount*Q_hat1[s_next, a_next]) - Q_hat2[s_now, a_now])  # updating action-value function

                # Updating epsilon-greedy policy
                b[s_now, :] = np.ones(A)*epsilon/A # assigning epsilon/A probability of selection to all actions
                greedy = np.argmax(Q_hat1[s_now, :]+Q_hat2[s_now, :]) # identifying greedy action in current state
                b[s_now, greedy] += (1-epsilon) # increasing the probability of selection of the best action in current state

                # Updating current state
                s_now = s_next

    # Combining estimates of action-value functions (accounting for possibly different number samples in Q_hat1 and Q_hat2)
    Q_hat = (N1*Q_hat1 + (N-N1)*Q_hat2)/N

    # Identifying approximately optimal policy
    pi = np.argmax(Q_hat, axis=1)

    return Q_hat, pi


# Example
## Parameters
r = np.array([[10, 5], [0, 0]]) # rewards
Pd1 = np.array([[0.5, 0.5], [0, 1]]); Pd2 = np.array([[0, 1], [0, 1]]); P = np.dstack((Pd1, Pd2)) # transition probabilities
epsilon = 0.1; discount = 0.95; N = 500; s_term = 1 # algorithmic parameters
pi_eval = np.array([[1, 0], [0, 1]]) # policy of interest (for policy evaluation algorithms)

## Solving MDP
Q, v, pi = policy_iteration(P, r, 0, discount) # added for comparison purposes
v_hat = td_0(pi_eval, P, r, s_term, discount, N) # TD(0) algorithm
Q_hat_s, pi_s = sarsa(P, r, s_term, discount, N) # Sarsa algortihm
Q_hat_ql, pi_ql = q_learning(P, r, s_term, discount, N, epsilon) # Q-learning algorithm
Q_hat_es, pi_es = expected_sarsa(P, r, s_term, discount, N) # Expected Sarsa algorithm
Q_hat_dql, pi_dql = double_q_learning(P, r, s_term, discount, N, epsilon) # Double Q-learning algorithm
