#!/bin/python3


import numpy as np
from utils import *
from utils_deep import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


def test_qlearning():
    # Paramètres
    ALPHA = 0.9
    GAMMA = 0.5
    EPSILON_START = .99
    EPOCHS = 10000

    space = initialize_space()

    #=== TEST 1 : R = 30, -10, -1, -2 ===
    R1 = (2, -2, -1, 0)  # (goal, dragon, out_of_bounds, empty)
    Q1 = train_Q_learning(ALPHA, GAMMA, EPSILON_START, EPOCHS, R1)

    print("Q-table shape:", Q1.shape)
    print("Q1 : \n", Q1)
    print("Training completed!")

    #Test the learned policy
    test_policy(Q1, R1)  # Version console


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


# ======================
#       MAIN
# ======================
if __name__ == "__main__":
    # test_qlearning()
    test_qdeeplearning()

    pass
    

