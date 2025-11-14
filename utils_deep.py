from utils import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from typing import Tuple
import time
import tensorflow as tf
import random
from tqdm import tqdm
import keras
import matplotlib.pyplot as plt





def space_to_vec(position : tuple, N : int) -> np.ndarray:
    """
    Convertit une position (y,x) en vecteur one-hot aplati de taille N.

    Args:
        position (tuple): (y,x)
        N (int): nombre total d'états (ex: 16 pour 4x4)

    Returns:
        np.ndarray: vecteur de forme (1, N) one-hot
    """
    vec_etat = np.zeros((1, N), dtype=float)
    # largeur du plateau (supposons N carré, ex: 16 -> 4)
    width = int(np.sqrt(N)) if N > 0 else 1
    idx = int(position[0] * width + position[1])
    vec_etat[0, idx] = 1.0
    return vec_etat


def build_dnn_model(input_shape: Tuple[int] = (16,)) -> Model:
    """
    Crée un modèle de Deep Q-Network (DQN) avec l'architecture suivante:
    Input(16) → Dense(128) → ReLU → Dense(128) → ReLU → Dense(4)
    
    Args:
        input_shape (Tuple[int]): Forme de l'entrée, par défaut (16,) pour un plateau 4x4 aplati
    
    Returns:
        Model: Modèle Keras compilé prêt pour l'entraînement
    
    Notes:
        - Input de taille 16 (4x4 plateau aplati)
        - Output de taille 4 (une Q-value pour chaque action possible)
        - Utilise ReLU comme fonction d'activation
        - Optimiseur Adam avec MSE comme fonction de perte
    """
    model = Sequential([
        Input(shape=input_shape),             
        Dense(128),                           
        Activation('relu'),                   
        Dense(128),                           
        Activation('relu'),                   
        Dense(4)                              
    ])
    
    # Compilation du modèle
    model.compile(
        optimizer=Adam(learning_rate=0.1),
        loss=MeanSquaredError()
    )
    
    model.summary()
    return model






def choose_action(vec_state : np.ndarray, epsilon: float, model ) -> int:
    """
    Choose an action according to the epsilon-greedy strategy.

    Args:
        position (tuple[int, int]): Current agent position (y, x)
        epsilon (float): Exploration probability (between 0 and 1). With
            probability epsilon, choose a random action,
            otherwise choose the exploitation action (best Q-value).
        model : DNN that predict the best action

    Returns:
    """
    

    if not (0 <= epsilon <= 1):
        raise ValueError(f"epsilon must be in [0,1], got {epsilon}")

    value = model.predict(vec_state, verbose=False)

    if random.uniform(0, 1) < epsilon:
        return (random.randint(0, 3), value) 
    else:
        return (np.argmax(value), value)



def train_Q_learning_DQN(alpha: float, gamma: float, epsilon_start: float, episode: int, rewards: tuple[int, int, int, int] ) -> np.ndarray:
    """
    Train a Deep Q-Learning model using a DNN.

    Args:
        alpha (float): Learning rate
        gamma (float): Discount factor for future rewards
        epsilon_start (float): Initial exploration probability
        episode (int): Number of training episodes
        rewards (tuple[int, int, int, int]): Rewards for (goal, step, obstacle, out_of_bounds)
    Returns:
        np.ndarray: Q-table learned by the model
    """

    model = build_dnn_model()


    
    epsilon = epsilon_start
    epsilon_min = 0.1
    epsilon_decay = 0.8

    

    suivi = np.zeros((episode))

    

    for i in tqdm(range(episode)):
        pos = (0, 0)
        space = initialize_space()

        N = space.shape[0]*space.shape[1]
        done = False

        while not done:
            vec_state = space_to_vec(pos, N)


            action, _ = choose_action(vec_state, epsilon, model)


            new_pos, reward, done = apply_action(action, pos, space, rewards)
            space[pos[0], pos[1]] = EMPTY
            space[new_pos[0], new_pos[1]] = PLAYER
            vec_new_state = space_to_vec(new_pos, N)
            target_q = reward + gamma * model.predict(vec_new_state, verbose=False).max()
            target_q = tf.convert_to_tensor(target_q)  

            # Calcul différentiable à l'intérieur du tape
            with tf.GradientTape() as tape:
                q_values = model(vec_state)  
                predicted_q = q_values[0, action]  
                loss = keras.losses.mean_squared_error([target_q], [predicted_q])

            # Appliquer gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            suivi[i] += loss.numpy()
            # print(loss)

            pos = new_pos





        epsilon = max(epsilon_min, epsilon * epsilon_decay)  

    plt.plot(suivi)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss over Episodes')
    plt.savefig('training_loss.png')
    # plt.show()

    return model


def test_policy_DQN(model, rewards: tuple[int, int, int, int]) -> None:
    """
    Test the learned policy using the DQN model.

    Args:
        model : DNN that predict the best action
        rewards (tuple[int, int, int, int]): Rewards for (goal, step, obstacle, out_of_bounds)
    """
    steps = 0
    pos = (0, 0)
    space = initialize_space()
    N = space.shape[0] * space.shape[1]
    done = False
    total_reward = 0

    while not done:
        vec_state = space_to_vec(pos, N)
        action, _ = choose_action(vec_state, 0.0, model)  # Exploitation only


        new_pos, reward, done = apply_action(action, pos, space, rewards)
        space[pos[0], pos[1]] = EMPTY
        space[new_pos[0], new_pos[1]] = PLAYER

        total_reward += reward
        pos = new_pos
        steps += 1

        print(f"\nStep {steps}: Action {['Up', 'Right', 'Down', 'Left'][action]}")
        print(f"Position: {pos}")
        # time.sleep(0.5)

        if done:

            print(f"Victory! Total reward: {total_reward} in {steps} steps.")



    print("Total reward during test:", total_reward)
