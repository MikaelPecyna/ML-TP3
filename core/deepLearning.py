import time

import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from core.launcher import Launcher
import core.utils_deep as core_utils
from tqdm import tqdm
import matplotlib.pyplot as plt

class LauncherDL(Launcher):

    def __init__(self, width, height):
        self.all_moves = []
        self.max_epoch = 1000

        self.current_scores = {}   

        self.test_path = []
        self.test_total_reward = 0
        self.test_steps = 0
       
        self.epoch_scores = []  
        self.epoch_steps  = []
        self.model = None
        self.R1 = (100, -20, -3, -1)  # (goal, dragon, out_of_bounds, empty)


    def launch_training(self):
        # Paramètres
        ALPHA = 0.001
        GAMMA = 0.999
        EPSILON_START = .999
        EPOCHS = 1000

        #=== TEST 1 : R = 30, -10, -1, -2 ===
        self.model = self.train_Q_learning_DQN(ALPHA, GAMMA, EPSILON_START, EPOCHS, self.R1)

        print("DNN model trained!")
        print(self.model.summary())
        
    def train_Q_learning_DQN(self, alpha: float, gamma: float, epsilon_start: float, episode: int, rewards: tuple[int, int, int, int] ) -> Model:
        """
        Train a Deep Q-Learning model using a DNN.

        Args:
            alpha (float): Learning rate
            gamma (float): Discount factor for future rewards
            epsilon_start (float): Initial exploration probability
            episode (int): Number of training episodes
            rewards (tuple[int, int, int, int]): Rewards for (goal, step, obstacle, out_of_bounds)
        Returns:
            Model : Trained DNN model
        """
        model = core_utils.build_dnn_model()

        start_time = time.time()
        epsilon = epsilon_start
        epsilon_min = 0.1
        epsilon_decay = 0.995

        suivi = np.zeros((episode))

        pbar = tqdm(range(episode), desc="Training Episodes", unit="ep")
        for i in pbar:
            pos = (0, 0)
            space = core_utils.initialize_space()

            N = space.shape[0]*space.shape[1]
            done = False
            steps = 0

            while not done:
                vec_state = core_utils.space_to_vec(pos, N)
                action, _ = core_utils.choose_action(vec_state, epsilon, model)

                new_pos, reward, done = core_utils.apply_action(action, pos, space, rewards)
                space[pos[0], pos[1]] = core_utils.EMPTY
                space[new_pos[0], new_pos[1]] = core_utils.PLAYER
                vec_new_state = core_utils.space_to_vec(new_pos, N)
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
                steps += 1

            pbar.set_postfix({
                'Episode': i+1,
                'Epsilon': f"{epsilon:.3f}",
                'Avg Loss': f"{np.mean(suivi[:i+1]):.4f}",
                'Steps/Ep': steps,
                'Time': f"{time.time() - start_time:.1f}s"
            })
            epsilon = max(epsilon_min, epsilon * epsilon_decay)  

        plt.plot(suivi)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss over Episodes')
        plt.savefig('training_loss.png')
        # plt.show()

        return model
    
    def launch_test(self):
        #Test the learned policy
        core_utils.test_policy_DQN(self.model, self.R1) 