#!/bin/python3

import numpy as np
from core.launcher import Launcher
from core.utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants representing different game elements
EMPTY   = 0  # Empty cell on the space
PLAYER  = 1  # Player position
GOAL    = 2  # Goal cell
DRAGON  = 3  # Dragon position

class LauncherQL(Launcher):
    def __init__(self, width, height):
        self.all_moves = []
        self.max_epoch = 1000

        self.test_path = []
        self.test_total_reward = 0
        self.test_steps = 0
        
        self.current_scores = {}   
       
        self.epoch_scores = []  
        self.epoch_steps  = []  
        self.R1 = (40, -20, -2, -1)  # (goal, dragon, out_of_bounds, empty)
        self.Q1 = None

    def launch_training(self):
        # Paramètres
        ALPHA = 0.9
        GAMMA = 0.5
        EPSILON_START = .99
     
        #space = initialize_space()

        #=== TEST 1 : R = 30, -10, -1, -2 ===
        self.Q1 = self.train_Q_learning(ALPHA, GAMMA, EPSILON_START, self.max_epoch, self.R1)

        print("Q-table shape:", self.Q1.shape)
        print("Q1 : \n", self.Q1)
        print("Training completed!")

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

        for epoch in tqdm(range(episode)):
            pos = (0, 0)
            space = initialize_space()
            done = False
            self.current_scores = []
            #epsilon =  1.0 - (epoch / episode) 

            while not done:
                state = state_to_index(pos)
                action = choose_action(pos, epsilon, Q)
                new_pos, reward, done = apply_action(action, pos, space, rewards)

                # ← ENREGISTREMENT DU MOVE :
                self.all_moves.append((pos, action, new_pos))

                # ACCUMULATION DU SCORE
                self.current_scores.append(reward)

                next_state = state_to_index(new_pos)
                update_Q_table(Q, state, next_state, action, alpha, reward, gamma, done)
                
                pos = new_pos
         
            # ENREGISTREMENT DES STATS DE L'ÉPISODE
            self.epoch_steps.append(epoch)
            self.epoch_scores.append(self.current_scores)

            # DECROISSANCE D'EPSILON
            epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Reduce epsilon each episode => Less exploration

        return Q

    def launch_test(self):
        if self.Q1 is None:
            print("Error: Q-table is not trained yet.")
            return
        #Test the learned policy
        self.test_policy(self.Q1, self.R1)
        return

    def test_policy(self,Q: np.ndarray, rewards: tuple[int, int, int, int]):
        """
        Test the learned policy by running an episode.

        Args:
            Q (np.ndarray): Trained Q-table
            rewards (tuple[int, int, int, int]): Reward structure

        Returns:
            tuple[list[tuple[int, int]], int, int]: Path taken, total reward, number of steps
        """
        import time

        pos = (0, 0)
        space = initialize_space()
        total_reward = 0
        steps = 0
        path = [pos]

        while True:
            state = state_to_index(pos)
            action = choose_action(pos, epsilon=0.0, Q=Q)  # Optimal policy
            new_pos, reward, done = apply_action(action, pos, space, rewards)

            # Update space
            space[pos[0], pos[1]] = EMPTY
            space[new_pos[0], new_pos[1]] = PLAYER

            total_reward += reward
            steps += 1
            path.append(new_pos)
            pos = new_pos

            print(f"\nStep {steps}: Action {['Up', 'Right', 'Down', 'Left'][action]}")
            print(f"Position: {pos}")
            # time.sleep(0.5)

            if done:
                if reward > 0:
                    print(f"Victory! Total reward: {total_reward} in {steps} steps.")
                else:
                    print(f"Failure (dragon or out of bounds). Total reward: {total_reward}")
                break
        self.test_path = path
        self.test_total_reward = total_reward
        self.test_steps = steps
        return 



    

