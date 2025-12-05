import numpy as np
import random
from tqdm import tqdm

# Constants representing different game elements
EMPTY   = 0  # Empty cell on the space
PLAYER  = 1  # Player position
GOAL    = 2  # Goal cell
DRAGON  = 3  # Dragon position


def initialize_space(height: int = 4, width: int = 4) -> np.ndarray:
    """
    Initialize and return the game space with the initial configuration.

    The space is represented by a 2D grid where:
    - EMPTY (0)   : Accessible empty cell
    - PLAYER (1)  : Current player position
    - GOAL (2)    : Goal cell to reach
    - DRAGON (3)  : Cell occupied by a dragon to avoid

    Args:
        height (int): Number of rows in the space (default: 4)
        width (int): Number of columns in the space (default: 4)

    Returns:
        np.ndarray: 2D NumPy array with the initial game configuration:
                    - Player at position (0,0)
                    - Dragons at positions (1,0), (1,2), (2,3), (3,1)
                    - Goal at position (3,3)
                    Note: Access via space[y][x]
    """
    # Create an empty space
    space = np.zeros((height, width), dtype=int)

    # Place initial elements
    space[0][0] = PLAYER    # Player starting position
    space[0][1] = DRAGON    # First dragon
    space[2][1] = DRAGON    # Second dragon
    space[3][2] = DRAGON    # Third dragon
    space[1][3] = DRAGON    # Fourth dragon
    space[3][3] = GOAL      # Goal position

    return space


def initialize_space_random(height: int = 4, width: int = 4, num_dragons: int = 4) -> np.ndarray:
    """
    Initialize and return a random game space configuration.

    The space is represented by a 2D grid where:
    - EMPTY (0)   : Accessible empty cell
    - PLAYER (1)  : Current player position
    - GOAL (2)    : Goal cell to reach
    - DRAGON (3)  : Cell occupied by a dragon to avoid

    Args:
        height (int): Number of rows in the space (default: 4)
        width (int): Number of columns in the space (default: 4)
        num_dragons (int): Number of dragons to place (default: 4)

    Returns:
        np.ndarray: 2D NumPy array with a random game configuration:
                    - Player at position (0,0)
                    - Goal at position (height-1, width-1)
                    - Dragons randomly placed (not on player or goal positions)
                    Note: Access via space[y][x]
    """
    # Create an empty space
    space = np.zeros((height, width), dtype=int)

    # Place player at starting position
    space[0][0] = PLAYER

    # Place goal at bottom-right corner
    space[height-1][width-1] = GOAL

    # Generate list of all possible positions except (0,0) and (height-1,width-1)
    all_positions = [(i, j) for i in range(height) for j in range(width)
                     if not ((i == 0 and j == 0) or (i == height-1 and j == width-1))]

    # Randomly select positions for dragons
    dragon_positions = random.sample(all_positions, min(num_dragons, len(all_positions)))

    # Place dragons
    for pos in dragon_positions:
        space[pos[0]][pos[1]] = DRAGON

    return space




def get_goal_position(space: np.ndarray) -> tuple[int, int]:
    """
    Return the position of the goal cell on the space.

    Args:
        space (np.ndarray): Current state of the game space

    Returns:
        tuple[int, int]: A tuple containing (y, x) where:
            - y is the vertical position (row)
            - x is the horizontal position (column)
            Note: Coordinates start at 0
    """
    y, x = np.where(space == GOAL)
    return (int(y[0]), int(x[0]))



def apply_action(action: int, position: tuple[int, int], space: np.ndarray, rewards: tuple[int, int, int, int]) -> tuple[tuple[int, int], int, bool]:
    """
    Evaluate the result of an agent's action in the environment and return the new state.
    This function is used in the context of reinforcement learning.

    Args:
        action (int): The action chosen by the agent:
            0: Up        (+Y)
            1: Right     (+X)
            2: Down      (-Y)
            3: Left      (-X)
        position (tuple[int, int]): Current agent position (y, x)
        space (np.ndarray): Current state of the game space
        rewards (tuple[int, int, int, int]): Tuple of rewards:
            rewards[0]: Reward for reaching the goal
            rewards[1]: Penalty for encountering a dragon
            rewards[2]: Penalty for going out of bounds
            rewards[3]: Penalty for empty cell (normal move)

    Returns:
        tuple[tuple[int, int], int, bool]: A tuple containing:
            - The new position (y, x) after the action
            - The reward obtained based on the situation
            - A boolean indicating if the episode is terminated (victory or failure)

    Note:
        - Coordinates are in (y, x) where y increases upwards
        - If the action leads out of the space, the agent stays in place
    """
    # Current position (y increases upwards)
    pos_y, pos_x = position[0], position[1]
    space_height, space_width = space.shape
    
    # Calculate new position based on action
    new_y, new_x = pos_y, pos_x
    if action == 0:      # Up (+Y)
        new_y += 1
    elif action == 1:    # Right (+X)
        new_x += 1
    elif action == 2:    # Down (-Y)
        new_y -= 1
    elif action == 3:    # Left (-X)
        new_x -= 1
    
    # Check space boundaries
    if not (0 <= new_x < space_width and 0 <= new_y < space_height):
        # If out of bounds, stay in place with penalty
        return (pos_y, pos_x), rewards[2], False
    
    # Check the content of the new cell
    cell_content = space[new_y, new_x]

    if cell_content == DRAGON:
        # Collision with dragon: penalty and reset to start
        return (0, 0), rewards[1], True
    elif cell_content == GOAL:
        # Reached goal: reward and victory
        return (new_y, new_x), rewards[0], True
    else:
        # Normal move: movement penalty
        return (new_y, new_x), rewards[3], False


def choose_action(position: tuple[int, int], epsilon: float, Q: np.ndarray) -> int:
    """
    Choose an action according to the epsilon-greedy strategy.

    Args:
        position (tuple[int, int]): Current agent position (y, x)
        epsilon (float): Exploration probability (between 0 and 1). With
            probability epsilon, choose a random action,
            otherwise choose the exploitation action (best Q-value).
        Q (np.ndarray): Q-table of expected rewards

    Returns:
        int: Index of the chosen action (0=Up, 1=Right, 2=Down, 3=Left)

    Notes:
        - In case of ties between multiple actions during exploitation,
            the action is chosen randomly among the best ones.
    """
    state = state_to_index(position)

    if not (0 <= epsilon <= 1):
        raise ValueError(f"epsilon must be in [0,1], got {epsilon}")


    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # Exploration
    else:
        # Exploitation with random tie-breaking
        q_values = Q[state]  # [q0, q1, q2, q3]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]  # All indices with max
        return int(random.choice(best_actions))
        

def state_to_index(position: tuple[int, int], space_width: int = 4) -> int:
    """
    Convert a position (y,x) to a unique index for the Q-table.

    Args:
        position (tuple[int, int]): Current position (y, x) on the space
        space_width (int): Width of the space (default: 4)

    Returns:
        int: Unique index of the state in the Q-table

    Notes:
        - Uses row-major encoding:
          idx = y * space_width + x
        - For a 4x4 space, indices range from 0 to 15
    """
    y, x = position
    return y * space_width + x


    


def initialize_Q_table(space: np.ndarray) -> np.ndarray:
    """
    Initialize the Q-table to zero for all states and actions.
    Args:
        space: np.ndarray of shape (height, width)
    Returns:
        Q: np.ndarray of shape (num_states, 4)
    """
    height, width = space.shape
    num_states = height * width
    num_actions = 4
    Q = np.zeros((num_states, num_actions))
    return Q





def update_Q_table(Q: np.ndarray, state_idx: int, next_state_idx: int, action: int, alpha: float, reward: float, gamma: float, done: bool) -> np.ndarray:
    """
    Update the Q-table using the Q-learning update rule.

    Args:
        Q (np.ndarray): Q-table to update
        state_idx (int): Current state index
        next_state_idx (int): Next state index
        action (int): Action taken
        alpha (float): Learning rate
        reward (float): Reward received
        gamma (float): Discount factor
        done (bool): Whether the episode is done

    Returns:
        np.ndarray: Updated Q-table
    """
    if done:
        max_next = 0
    else:
        max_next = np.max(Q[next_state_idx])

    Q[state_idx][action] += alpha * (reward + gamma * max_next - Q[state_idx][action])
    return Q

    #  Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))

def train_Q_learning(alpha: float, gamma: float, epsilon_start: float, episode: int, rewards: tuple[int, int, int, int] ) -> np.ndarray:
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
            next_state = state_to_index(new_pos)
            update_Q_table(Q, state, next_state, action, alpha, reward, gamma, done)
            pos = new_pos

        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Reduce epsilon each episode => Less exploration

    return Q




