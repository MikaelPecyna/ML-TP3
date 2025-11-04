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



def space_to_vec(space : np.ndarray) -> np.ndarray:
    """
    Convertit l'état du plateau de jeu en un vecteur d'entrée pour le modèle DNN.
    
    Args:
        space (np.ndarray): État du plateau de jeu sous forme de matrice 4x4
    
    Returns:
        np.ndarray: Vecteur aplati de taille 16 représentant l'état du plateau
    """
    return space.flatten()


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
        int: Index of the chosen action (0=Up, 1=Right, 2=Down, 3=Left)

    """
    

    if not (0 <= epsilon <= 1):
        raise ValueError(f"epsilon must be in [0,1], got {epsilon}")


    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  
    else:
        value = model.predict(np.array([vec_state]))[0]
        return np.argmax(value)


def update_Q_DQN(batch : list, model, alpha: float, gamma: float) -> None:
    """
    Met à jour les poids du modèle DQN en utilisant un batch d'expériences en utlisant tf.GradientTape.

    Args:
        batch (list): Liste de tuples (state, action, reward, next_state, done
        model : DNN that predict the best action
        alpha (float): Learning rate
        gamma (float): Discount factor for future rewards
    """

    """
    Note perso 


    y_j = { r_j /\ r_j + y max_a' Q(state_j, a_j, theta (????))} [1] -> Terminal j+1 ; [2] -> Non-terminal j+2

    gradient descent (y_j - Q(state_j, a_j, theta))^2


    """

    state, action, reward, vec_next_state, done = [batch]

    with tf.GradientTape as tape : 
        if(done) :
            loss = reward
        else : 
            value = model.predict(np.array([vec_next_state]))[0]
            qmax = np.argmax(value)
            loss = reward + gamma * qmax
        
        
        
    


    

    



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
    epsilon_decay = 0.995

    for _ in tqdm(range(episode)):
        pos = (0, 0)
        space = initialize_space()
        done = False

        while not done:
            old_space = space.copy()
            action = choose_action(vec_state, epsilon, model)

            new_pos, reward, done = apply_action(action, pos,  space, rewards)
            space[pos[0], pos[1]] = EMPTY
            space[new_pos[0], new_pos[1]] = PLAYER

            vec_state = space_to_vec(old_space)

            vec_next_state = space_to_vec(space)



            batch = [(vec_state, action, reward, vec_next_state, done)]
            update_Q_DQN(batch, model, alpha, gamma)

            pos = new_pos


        epsilon = max(epsilon_min, epsilon * epsilon_decay)  

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
    done = False
    total_reward = 0

    while not done:
        vec_state = space_to_vec(space)
        action = choose_action(vec_state, 0.0, model)  # Exploitation only
        

        new_pos, reward, done = apply_action(action, pos, space, rewards)
        space[pos[0], pos[1]] = EMPTY
        space[new_pos[0], new_pos[1]] = PLAYER

        total_reward += reward
        pos = new_pos

        print(f"\nStep {steps}: Action {['Up', 'Right', 'Down', 'Left'][action]}")
        print(f"Position: {pos}")
        time.sleep(0.5)

        if done:
            if reward > 0:
                print(f"Victory! Total reward: {total_reward} in {steps} steps.")
            else:
                print(f"Failure (dragon or out of bounds). Total reward: {total_reward}")
            break

    print("Total reward during test:", total_reward)