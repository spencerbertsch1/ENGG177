"""
--------------------------------------
| Decision Making Under Unvertainty  |
| ENGG 177 - Homework #3             |
| Dartmouth College                  |  
| Spring 2022                        |
| Spencer Bertsch                    |
--------------------------------------

This file contains the code needed to build an environment representing a 4 state MDP and find an 
approximately optimal policy using an on-policy first-visit MC control algorithm. 

This file can be run from the command line by running: $ python3 first_visit_mc.py
"""

import numpy as np
import random

class Env():

    def __init__(self):
        self.transition_matrix = self.create_transition_matrix()
        self.state = random.choice([1, 2, 3, 4])  # <-- initialize to random state in the system
        self.action = True  # <-- here True represents "stay", False represents "leave"
        self.discount_rate = 0.9


    def check_env(self):
        """
        Small utility function to check that the transition function is working properly
        """
        for i in range(50):
            env.update_state()
            print(f' ------------------------- NEW STATE = {env.state} ------------------------- ')


    def create_transition_matrix(self) -> np.array:
        """
        Utility function that returns the transition probability matrix
        """
        mat: np.array = np.array([[0.3, 0.4, 0.2, 0.1],  
                                  [0.2, 0.3, 0.5, 0.0], 
                                  [0.1, 0.0, 0.8, 0.1], 
                                  [0.4, 0.0, 0.0, 0.6]], np.float64)

        return mat


    def step_using_probs(self) -> int:
        """
        This is a utility function that returns the new state given the step is made because the action was 
        a_1 - to choose to stay inside the system. 

        This function does NOT take the U probability function into account, it simply returns a step given the 
        probability matrix denoted here: self.transition_matrix

        :return: int representing the new state
        """
        curr_state = self.state.copy()
        t_mat: np.array = self.transition_matrix.copy()

        # get the row of transition probabilities for the current state 
        prob_list: list = list(t_mat[(curr_state-1), :])

        # round the elements of the list 
        prob_list = [round(prob, 2) for prob in prob_list]

        # define the new state given the probabilities from the transition matrix
        new_state = np.random.choice([1, 2, 3, 4], p=prob_list)      

        return new_state


    def update_state(self):
        """
        A function that can be used to get the next state given a current state and an action
        """

        if self.action:

            # define the value of U drawn from a uniform distribution between 0 and 1, [0, 1]. 
            U = np.random.uniform(1,0)
            t_mat: np.array = self.transition_matrix.copy()
            si: int = self.state

            # we now need to find p(s_1|s_i, a_1), which at this point is p(s_1|s_i) because we know a = a_1
            prob_s1_given_si: float = t_mat[(si-1), 1]

            if U <= prob_s1_given_si:
                # print(f'U is {round(U, 3)} which is less than the probability of s1 given si: {prob_s1_given_si}, setting state to 1.')
                self.state = 1 

            else:
                # define each of the possible next states as u
                for k in [2, 3, 4]:

                    # for each k value, we need to find the state s_k that we will transition to 

                    # find the sum of probabilities of transitioning from state si to states sj up to value k-1
                    sj_prob_sum_to_k_minus_one = 0
                    for sj in range(1, k-1, 1):
                        # we now need to find p(s_j|s_i)
                        prob_sj_given_si: float = t_mat[(si-1), (sj-1)]
                        sj_prob_sum_to_k_minus_one = sj_prob_sum_to_k_minus_one + prob_sj_given_si

                    # find the sum of probabilities of transitioning from state si to states sj up to value k
                    sj_prob_sum_to_k = 0
                    for sj in range(1, k, 1):
                        # we now need to find p(s_j|s_i)
                        prob_sj_given_si: float = t_mat[(si-1), (sj-1)]
                        sj_prob_sum_to_k = sj_prob_sum_to_k + prob_sj_given_si

                    if (sj_prob_sum_to_k_minus_one < U) & (U <= sj_prob_sum_to_k):
                        self.state = k

        else:
            # here we set the state to 20 - an absorbing state, the game is over
            self.state = 20

    
    def get_next_action(self):
        """
        Get the next action given the current state

        s_t --> a_t
        """

        # TODO here we need to implement the logic that uses the policy (pi) to get the next action! 
        next_action: bool = True

        self.action = next_action



    def first_visit_mc(self):
        """
        First visit MC algorithm implementation
        """

        # initialize parameters: 
        Ns: int = 0
        N: int= 500 # <-- later set to 500 ! 
        gs: dict = {1: 0, 2: 0, 3: 0, 4: 0}
        v_hat_pi: dict = {1: 0, 2: 0, 3: 0, 4: 0}

        # what should the value of T be? 
        T = 50

        for n in range(N):

            visited_states: dict = {1: False, 2: False, 3: False, 4: False}

            # we move forward from t=0 to t=T to generate the episode data
            episode_results: dict = {}
            for t in range(T):
                reward = self.state
                episode_results[t] = [self.state, self.action, reward]

            G = 0
            # we now move backward from t=T to t=0 to find all the problem parameters
            for t in range(T-1, 0, -1):
                # get the reward from the episode results
                state_t = episode_results[t][0]
                action_t = episode_results[t][1]
                reward_t = episode_results[t][2]
                
                # update G
                G = (self.discount_rate * G) + reward_t

                if visited_states[state_t] is False: 
                    # this is the first time we have visited this state in this episode! 

                    # update Ns and gs
                    Ns = Ns + 1
                    gs[state_t] = gs[state_t] + G

                    # update v_hat of state s
                    v_hat_pi[state_t] = gs[state_t]/Ns

                    # set this state to visited
                    visited_states[state_t] = True

        print(v_hat_pi)

# some test code
if __name__ == "__main__":
    env = Env()
    env.first_visit_mc()
