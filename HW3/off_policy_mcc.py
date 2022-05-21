"""
--------------------------------------
| Decision Making Under Unvertainty  |
| ENGG 177 - Homework #3             |
| Dartmouth College                  |  
| Spring 2022                        |
| Author: Spencer Bertsch            |
--------------------------------------

This file contains the code needed to build an environment representing a 5 state MDP and find an 
approximately optimal policy using an off-policy first-visit MC control algorithm. 

This file can be run from the command line by running: $ python3 first_visit_mc.py
"""

import numpy as np
import random
random.seed = 42

class Env():

    # define class variables
    verbose = True  # <-- set to true to see all the detailed outputs

    def __init__(self):
        self.transition_matrix = self.create_transition_matrix()
        self.state = random.choice([1, 2, 3, 4])  # <-- initialize to random state in the system
        self.action = 0  # <-- here 0 represents "stay", 1 represents "leave"
        self.lamda = 0.9  # <-- discount rate of 0.9
        self.epsilon = 1
        self.a_star = np.zeros(5, dtype=np.int32)  # <-- we initialize the 'best action' a_star to 0 or "stay"
        self.pi = np.full((5, 2), 0.5)


    def check_env(self):
        """
        Small utility function to check that the transition function is working properly
        """
        for i in range(50):
            env.get_next_state()
            s = '-'*20
            print(f' {s} NEW STATE = {env.state} {s} ')


    def create_transition_matrix(self) -> np.array:
        """
        Utility function that returns the transition probability matrix

        :return: nd.array representing the transition probability matrix 
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


    def get_next_state(self):
        """
        A function that can be used to get the next state given a current state and an action

        :return: NA - returns nothing, only updates instance variables
        """

        # here we want to sample pi to get the next action
        action_probabilities: list = list(self.pi[self.state-1]) 
        self.action = np.random.choice([0, 1], p=action_probabilities)

        if (self.action == 0) & (self.state != 5):

            # define the value of U drawn from a uniform distribution between 0 and 1, [0, 1]. 
            U = np.random.uniform(1,0)
            t_mat: np.array = self.transition_matrix.copy()
            si: int = self.state

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
                        break

        elif (self.action == 1) & (self.state != 5):
            # here we set the state to 5 - an absorbing state, the game is over
            self.state = 5
        else:
            if self.verbose:
                print('We\'re in state 5, continuing.')
            else:
                pass
    

    def on_policy_first_visit_mcc(self):
        """
        First visit MC algorithm implementation
        """

        # define the number of episodes
        N: int = 25 # <-- later set to 500 ! 

        """
        Rows: states
        Cols: Actions
        --------------
        |s1_a1, s1_a2|
        |s2_a1, s2_a2|
        |s3_a1, s3_a2|
        |s4_a1, s4_a2|
        |s5_a1, s5_a2|
        --------------
        """
        C_sa: np.array = np.full((5, 2), 0)
        Q_hat_sa: np.array = np.full((5, 2), 0)
        self.pi[:, 1] = np.argmax(Q_hat_sa, axis=1) 
        self.pi[:, 0] = 1 - self.pi[:, 1]

        N_sa: np.array = np.full((5, 2), 0)
        g_sa: np.array = np.full((5, 2), 0)

        for n in range(1, N, 1):
            if self.verbose:
                print(f'\n \n \n EPISODE #: {n}')
                print(f'Q_hat_sa: {Q_hat_sa}')

            # we move forward from t=0 to t=T to generate the episode data
            episode_results: dict = {}
            for t in range(1000000):
                self.get_next_state()
                action: bool = self.action
                state: int = self.state
                reward: int = self.state
                if state == 5:
                    reward = 20
                if self.verbose: 
                    print(f'Time: {t}, State: {state}, Action: {action}, Reward: {reward}')
                episode_results[t] = [state, action, reward]
                if state == 5:
                    if self.verbose:
                        print('Action \'leave\' was chosen so we\'re now in state 5. Stopping the episode.')
                    # we need to randomize the state before we start the next episode
                    self.state = np.random.choice([1, 2, 3, 4])
                    break

            W = 1
            G = 0
            if self.verbose: 
                print(f'Current Episode: {episode_results}')
            
            # we now move backward from t=T to t=0 to find all the problem parameters
            episode_length: int = max([k for k, _ in episode_results.items()])
            for t in range(episode_length, 0, -1):
                while W != 0: 

                    # get the reward from the episode results
                    state_t = episode_results[t][0]
                    action_t = episode_results[t][1]
                    reward_t = episode_results[t][2]
                    
                    # update G
                    G = (self.lamda * G) + reward_t

                    # update C_sa
                    C_sa[state_t-1][action_t] = C_sa[state_t-1][action_t] + W

                    # update Q_hat_sa
                    wcsa = W / C_sa[state_t-1][action_t]
                    gqsa = G - Q_hat_sa[state_t-1][action_t]
                    Q_hat_sa[state_t-1][action_t] = Q_hat_sa[state_t-1][action_t] + (wcsa * gqsa)

                    # update pi 
                    self.pi[:, 1] = np.argmax(Q_hat_sa, axis=1) 
                    self.pi[:, 0] = 1 - self.pi[:, 1]

                    if self.pi[state_t-1][0] == action_t:
                        continue
                    else:
                        # W = W*(1/(self.pi[state_t-1][action_t]))
                        W = W - 1


            if self.verbose: 
                print(f'New Policy: {self.pi}')
            
        print(f'Solution for On-Policy First Visit MCC Algorithm after {N} episodes:')
        print(f'Final Policy Pi: \n {np.around(self.pi, decimals=3)}')
        print(f'Final Q Hat: \n {np.around(Q_hat_sa, decimals=3)}')
        

# some test code
if __name__ == "__main__":
    env = Env()
    # env.check_env()
    env.on_policy_first_visit_mcc()
