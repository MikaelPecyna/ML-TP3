#!/bin/python3

import numpy as np
from core.utils import *
#from utils_deep import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


# Constants representing different game elements
EMPTY   = 0  # Empty cell on the space
PLAYER  = 1  # Player position
GOAL    = 2  # Goal cell
DRAGON  = 3  # Dragon position


class Launcher:
    def __init__(self, width, height):
        self.all_moves = []
        
    def test_qlearning(self):
        # Paramètres
        ALPHA = 0.9
        GAMMA = 0.5
        EPSILON_START = .99
        EPOCHS = 10000

        #space = initialize_space()

        #=== TEST 1 : R = 30, -10, -1, -2 ===
        R1 = (2, -2, -1, 0)  # (goal, dragon, out_of_bounds, empty)
        Q1 = self.train_Q_learning(ALPHA, GAMMA, EPSILON_START, EPOCHS, R1)

        print("Q-table shape:", Q1.shape)
        print("Q1 : \n", Q1)
        print("Training completed!")

        #Test the learned policy
        test_policy(Q1, R1)  # Version console

    def train_Q_learning(self, alpha: float, gamma: float, epsilon_start: float, episode: int, rewards: tuple[int, int, int, int] ) -> np.ndarray:
        """
        Train the Q-learning agent.

        Args:
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon_start (float): Initial epsilon for exploration
            episode (int): Number of training episodes
            rewards (tuple[int, int, int, int]): Reward structure (goal, dragon, out_of_bounds, step)

        Returns:
            np.ndarray: Trained Q-table
        """

        space = initialize_space()

        Q = initialize_Q_table(space)

        epsilon = epsilon_start
        epsilon_min = 0.1
        epsilon_decay = 0.995

        for _ in tqdm(range(episode)):
            pos = (0, 0)
            space = initialize_space()
            done = False

            while not done:
                state = state_to_index(pos)
                action = choose_action(pos, epsilon, Q)
                new_pos, reward, done = apply_action(action, pos, space, rewards)

                
                # ← ENREGISTREMENT DU MOVE :
                self.all_moves.append((pos, action, new_pos))

                next_state = state_to_index(new_pos)
                update_Q_table(Q, state, next_state, action, alpha, reward, gamma, done)
                pos = new_pos

            epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Reduce epsilon each episode => Less exploration

        return Q



def test_qdeeplearning():
    # Paramètres
    ALPHA = 0.001
    GAMMA = 0.5
    EPSILON_START = .99
    EPOCHS = 150

    space = initialize_space()

    #=== TEST 1 : R = 30, -10, -1, -2 ===
    R1 = (100, -20, -3, -1)  # (goal, dragon, out_of_bounds, empty)
    model = train_Q_learning_DQN(ALPHA, GAMMA, EPSILON_START, EPOCHS, R1)

    print("DNN model trained!")
    print(model.summary())

    #Test the learned policy
    test_policy_DQN(model, R1)  # Version console



    

