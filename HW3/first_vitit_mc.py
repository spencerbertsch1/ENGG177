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
        self.state = 1  # <-- initialize state to 1
        self.action = True  # <-- here True represents "stay", False represents "leave"

    def get_next_state(self):
        """
        A function that can be used to get the next state given a current state and an action

        p(s(t+1)|st, at)
        """

        if self.action:

            t_mat: np.array = self.transition_matrix.copy()

            # get the row of transition probabilities for the current state 
            prob_list: list = list(t_mat[(self.state-1), :])

            # round the elements of the list 
            prob_list = [round(prob, 2) for prob in prob_list]

            # define the new state given the probabilities from the transition matrix
            new_state = np.random.choice([1, 2, 3, 4], p=prob_list)      

            self.state = new_state

        else:
            # here we set the state to 0 - an absorbing state, the game is over
            self.state = 0



    def create_transition_matrix(self) -> np.array:
        """
        Utility function that returns the transition probability matrix
        """
        mat: np.array = np.array([[0.3, 0.4, 0.2, 0.1],  
                                  [0.2, 0.3, 0.5, 0.0], 
                                  [0.1, 0.0, 0.8, 0.1], 
                                  [0.4, 0.0, 0.0, 0.6]], np.float64)

        return mat


# some test code
if __name__ == "__main__":
    env = Env()

    for i in range(10):
        env.get_next_state()
