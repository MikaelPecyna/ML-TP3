import time
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from core.launcher import Launcher
import core.utils_deep as core_utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

class LauncherDDQN(Launcher):
    """
    Implémentation du Double Deep Q-Network (DDQN) qui utilise deux réseaux de neurones
    pour réduire le biais d'optimisme présent dans le DQN classique.
    
    L'algorithme DDQN utilise:
    - Online Network: pour sélectionner les actions (argmax)
    - Target Network: pour évaluer les valeurs Q
    - Mise à jour périodique du target network
    """

    def __init__(self, width, height):
        self.all_moves = []
        self.max_epoch = 1000

        self.current_scores = {}   

        self.test_path = []
        self.test_total_reward = 0
        self.test_steps = 0
       
        self.epoch_scores = []  
        self.epoch_steps  = []
        
        # Deux réseaux de neurones pour le Double DQN
        self.online_model = None      # Réseau principal pour la sélection d'actions
        self.target_model = None      # Réseau cible pour l'évaluation
        
        self.R1 = (100, -20, -3, -1)  # (goal, dragon, out_of_bounds, empty)
        
        # Paramètres spécifiques au DDQN
        self.update_frequency = 100   # Fréquence de mise à jour du target network
        self.update_counter = 0       # Compteur pour la mise à jour

    def launch_training(self):
        """
        Lance l'entraînement du modèle Double DQN.
        """
        # Paramètres d'entraînement
        ALPHA = 0.001
        GAMMA = 0.999
        EPSILON_START = 0.999
        EPOCHS = 1000

        # Initialisation des deux réseaux
        self.online_model = core_utils.build_dnn_model()
        self.target_model = core_utils.build_dnn_model()
        
        # Synchronisation initiale du target network avec l'online network
        self.update_target_network()

        print("Double DQN model initialized!")
        print("Online Network Summary:")
        print(self.online_model.summary())
        
        # Entraînement du modèle
        self.train_Double_DQN(ALPHA, GAMMA, EPSILON_START, EPOCHS, self.R1)

        print("Double DQN model trained!")
        
    def update_target_network(self):
        """
        Met à jour le target network avec les poids de l'online network.
        """
        self.target_model.set_weights(self.online_model.get_weights())
        print(f"Target network updated at episode {self.update_counter}")

    def train_Double_DQN(self, alpha: float, gamma: float, epsilon_start: float, 
                        episode: int, rewards: tuple[int, int, int, int]) -> Model:
        """
        Entraîne un modèle Double Deep Q-Network.

        Args:
            alpha (float): Taux d'apprentissage
            gamma (float): Facteur d'actualisation pour les récompenses futures
            epsilon_start (float): Probabilité initiale d'exploration
            episode (int): Nombre d'épisodes d'entraînement
            rewards (tuple[int, int, int, int]): Récompenses pour (goal, dragon, out_of_bounds, empty)
            
        Returns:
            Model: Modèle DNN entraîné
        """
        start_time = time.time()
        epsilon = epsilon_start
        epsilon_min = 0.1
        epsilon_decay = 0.995

        suivi = np.zeros((episode))
        losses = []

        pbar = tqdm(range(episode), desc="Training DDQN Episodes", unit="ep")
        for i in pbar:
            pos = (0, 0)
            space = core_utils.initialize_space()

            N = space.shape[0]*space.shape[1]
            done = False
            steps = 0
            episode_loss = 0

            while not done:
                vec_state = core_utils.space_to_vec(pos, N)
                action, _ = core_utils.choose_action(vec_state, epsilon, self.online_model)

                new_pos, reward, done = core_utils.apply_action(action, pos, space, rewards)
                space[pos[0], pos[1]] = core_utils.EMPTY
                space[new_pos[0], new_pos[1]] = core_utils.PLAYER
                vec_new_state = core_utils.space_to_vec(new_pos, N)
                
                # === ALGORITHME DOUBLE DQN ===
                
                # 1. L'online network choisit l'action
                online_q_values = self.online_model.predict(vec_state, verbose=False)
                best_action = np.argmax(online_q_values)
                
                # 2. Le target network évalue la valeur Q de cette action
                if done:
                    target_q = reward
                else:
                    target_q_values = self.target_model.predict(vec_new_state, verbose=False)
                    target_q = reward + gamma * target_q_values[0, best_action]
                
                # 3. Calcul de la perte et mise à jour
                with tf.GradientTape() as tape:
                    q_values = self.online_model(vec_state)
                    predicted_q = q_values[0, action]
                    loss = keras.losses.mean_squared_error([target_q], [predicted_q])

                # Appliquer les gradients
                gradients = tape.gradient(loss, self.online_model.trainable_variables)
                self.online_model.optimizer.apply_gradients(
                    zip(gradients, self.online_model.trainable_variables))
                
                episode_loss += loss.numpy()

                # 4. Mise à jour périodique du target network
                self.update_counter += 1
                if self.update_counter % self.update_frequency == 0:
                    self.update_target_network()

                pos = new_pos
                steps += 1

            # Enregistrement des statistiques
            losses.append(episode_loss)
            suivi[i] = episode_loss

            pbar.set_postfix({
                'Episode': i+1,
                'Epsilon': f"{epsilon:.3f}",
                'Avg Loss': f"{np.mean(losses[:i+1]):.4f}",
                'Steps/Ep': steps,
                'Target Updates': self.update_counter // self.update_frequency,
                'Time': f"{time.time() - start_time:.1f}s"
            })
            
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Graphique de la perte d'entraînement
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss over Episodes (DDQN)')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # Graphique comparatif de la perte cumulée
        cumulative_loss = np.cumsum(losses) / np.arange(1, len(losses) + 1)
        plt.plot(cumulative_loss)
        plt.xlabel('Episode')
        plt.ylabel('Average Loss')
        plt.title('Cumulative Average Loss (DDQN)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('ddqn_training_loss.png')
        print(f"Training plots saved as 'ddqn_training_loss.png'")

        return self.online_model
    
    def launch_test(self):
        """
        Teste la politique apprise par le modèle Double DQN.
        """
        if self.online_model is None:
            print("Error: Double DQN model is not trained yet.")
            return
            
        # Synchronisation finale du target network
        self.update_target_network()
        
        print("Testing Double DQN Policy...")
        self.test_policy_DDQN(self.online_model, self.target_model, self.R1)
    
    def test_policy_DDQN(self, online_model: Model, target_model: Model, 
                        rewards: tuple[int, int, int, int]) -> None:
        """
        Teste la politique apprise par le modèle Double DQN.

        Args:
            online_model (Model): Modèle online du DDQN
            target_model (Model): Modèle target du DDQN
            rewards (tuple[int, int, int, int]): Récompenses pour (goal, dragon, out_of_bounds, empty)
        """
        steps = 0
        pos = (0, 0)
        space = core_utils.initialize_space()
        N = space.shape[0] * space.shape[1]
        done = False
        total_reward = 0
        path = [pos]

        print("\n=== TEST DDQN ===")
        print(f"Starting position: {pos}")
        print(f"Goal position: (3, 3)")
        print(f"Rewards: {rewards}")

        while not done:
            vec_state = core_utils.space_to_vec(pos, N)
            
            # Utilisation du online model pour le test (politique gloutonne)
            action, q_values = core_utils.choose_action(vec_state, 0.0, online_model)
            
            new_pos, reward, done = core_utils.apply_action(action, pos, space, rewards)
            space[pos[0], pos[1]] = core_utils.EMPTY
            space[new_pos[0], new_pos[1]] = core_utils.PLAYER

            total_reward += reward
            pos = new_pos
            steps += 1
            path.append(pos)

            action_names = ['Up', 'Right', 'Down', 'Left']
            print(f"\nStep {steps}: Action {action_names[action]} (Q-values: {q_values[0]})")
            print(f"Position: {pos}, Reward: {reward}")

            if done:
                if reward > 0:
                    print(f"\n🎉 VICTORY! Total reward: {total_reward} in {steps} steps.")
                    print(f"Path taken: {path}")
                else:
                    print(f"\n❌ FAILURE (dragon or out of bounds). Total reward: {total_reward}")
                    print(f"Path taken: {path}")

        self.test_path = path
        self.test_total_reward = total_reward
        self.test_steps = steps
        
        # Analyse comparative des Q-values
        print(f"\n=== DDQN ANALYSIS ===")
        print(f"Total steps: {steps}")
        print(f"Total reward: {total_reward}")
        print(f"Average reward per step: {total_reward/steps:.3f}")
        
        return
    
    def compare_policies(self, dqn_model: Model = None) -> None:
        """
        Compare les performances du DDQN avec un DQN classique (optionnel).
        
        Args:
            dqn_model (Model): Modèle DQN classique pour comparaison
        """
        if dqn_model is not None:
            print("\n=== COMPARISON DDQN vs DQN ===")
            
            # Test DDQN
            self.update_target_network()
            print("Testing DDQN...")
            ddqn_reward, ddqn_steps = self._run_test_episode(self.online_model, self.target_model)
            
            # Test DQN
            print("Testing DQN...")
            dqn_reward, dqn_steps = self._run_test_episode(dqn_model, dqn_model)
            
            print(f"DDQN - Reward: {ddqn_reward}, Steps: {ddqn_steps}")
            print(f"DQN  - Reward: {dqn_reward}, Steps: {dqn_steps}")
            
            if ddqn_reward > dqn_reward:
                print("🏆 DDQN performs better!")
            elif ddqn_reward < dqn_reward:
                print("🏆 DQN performs better!")
            else:
                print("🤝 Equal performance!")
    
    def _run_test_episode(self, online_model: Model, target_model: Model) -> tuple[int, int]:
        """
        Exécute un épisode de test et retourne la récompense totale et le nombre d'étapes.
        
        Args:
            online_model (Model): Modèle online
            target_model (Model): Modèle target
            
        Returns:
            tuple[int, int]: (récompense_totale, nombre_étapes)
        """
        steps = 0
        pos = (0, 0)
        space = core_utils.initialize_space()
        N = space.shape[0] * space.shape[1]
        done = False
        total_reward = 0

        while not done:
            vec_state = core_utils.space_to_vec(pos, N)
            action, _ = core_utils.choose_action(vec_state, 0.0, online_model)
            new_pos, reward, done = core_utils.apply_action(action, pos, space, self.R1)
            
            total_reward += reward
            pos = new_pos
            steps += 1

            if steps > 50:  # Limite de sécurité
                break

        return total_reward, steps