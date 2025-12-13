# ML-TP3: Reinforcement Learning with Deep Q-Networks

A comprehensive reinforcement learning project implementing various Q-learning algorithms including classical Q-learning, Deep Q-Network (DQN), and Double Deep Q-Network (DDQN) in a grid-based game environment.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)


## 🎯 Project Overview

This project demonstrates reinforcement learning techniques through a grid-based navigation game where an agent learns to reach a goal while avoiding obstacles (dragons). The environment supports three different learning algorithms:

- **Q-Learning**: Classical table-based reinforcement learning
- **DQN**: Deep Q-Network with neural networks
- **DDQN**: Double Deep Q-Network with improved stability

## 🏗️ Project Structure

```
ML-TP3/
├── core/                          # Core RL algorithms
│   ├── qlearning.py              # Classical Q-Learning implementation
│   ├── deepLearning.py           # Deep Q-Network (DQN) implementation
│   ├── doubleqdeeplearning.py    # Double Deep Q-Network (DDQN) ⭐ NEW
│   ├── utils.py                  # General utilities
│   ├── utils_deep.py             # Deep learning utilities
│   └── launcher.py               # Abstract launcher class
├── view/                         # Visualization and UI
│   ├── engine.py                 # Pygame-based game engine
│   ├── animatedSprite.py         # Sprite animation handling
│   └── utils.py                  # UI utilities
├── sprites/                      # Game assets
│   ├── terrains/                 # Terrain textures
│   ├── perso/                    # Character sprites
│   └── flag/                     # Goal flag sprites
├── rapport/                      # Project report (LaTeX)
├── demo_ddqn.py                  # DDQN demonstration script ⭐ NEW
├── main.py                       # Main entry point
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow numpy pygame tqdm matplotlib
```

### Basic Usage

Run the main application with different algorithms:

```bash
# Classical Q-Learning
python3 main.py -op ql

# Deep Q-Network (DQN)
python3 main.py -op dl

# Double Deep Q-Network (DDQN) 
python3 main.py -op ddqn
```

## 🎮 Game Environment

### Environment Description

The game takes place on a 4x4 grid with:
- **Player** (blue robot): Starting at position (0,0)
- **Goal** (purple flag): Located at position (3,3)
- **Dragons** (slimes): Obstacles at positions (0,1), (2,1), (1,3), (3,2)
- **Empty cells**: traversable terrain

### Actions

The agent can perform 4 actions:
- `0`: Move Up
- `1`: Move Right  
- `2`: Move Down
- `3`: Move Left

### Rewards

Default reward structure:
- **Goal reached**: +100 points
- **Dragon encountered**: -20 points
- **Out of bounds**: -3 points
- **Normal move**: -1 point

### Controls

- `SPACE`: Start training
- `DOWN ARROW`: Test learned policy
- `LEFT/RIGHT ARROW`: Adjust animation speed

## 🧠 Algorithms Implemented

### 1. Q-Learning (Classical)

Table-based reinforcement learning algorithm.

**Key Features:**
- Q-table of size 16×4 (states × actions)
- Epsilon-greedy exploration strategy
- Direct Q-value updates

**Usage:**
```bash
python3 main.py -op ql
```

### 2. Deep Q-Network (DQN) ⭐

Neural network-based approach for continuous state spaces.

**Architecture:**
```
Input (16) → Dense(128, ReLU) → Dense(128, ReLU) → Dense(4)
```

**Key Features:**
- Experience replay for stable learning
- Target network for temporal difference targets
- Epsilon-greedy exploration with decay

**Usage:**
```bash
python3 main.py -op dl
```

### 3. Double Deep Q-Network (DDQN) ⭐ NEW

Enhanced version of DQN that reduces overestimation bias.

**Architecture:**
- **Online Network**: Selects actions (argmax)
- **Target Network**: Evaluates Q-values
- **Periodic Synchronization**: Updates target network weights

**Algorithm Flow:**
1. Online network chooses action: `action = argmax(Q_online(state))`
2. Target network evaluates: `Q_target = reward + γ * Q_target(state_next, best_action)`
3. Update online network with target Q-value
4. Periodically sync target network with online network

**Advantages over DQN:**
- Reduced overestimation bias
- Improved training stability
- Better convergence properties

**Usage:**
```bash
python3 main.py -op ddqn
```

## 📊 Performance Monitoring

### Training Metrics

The system provides comprehensive monitoring:

- **Episode progress** with tqdm visualization
- **Loss curves** saved as PNG files
- **Epsilon decay** tracking
- **Training time** measurement
- **Target network update frequency**

```

## 🔧 Configuration

### Hyperparameters

**Q-Learning:**
```python
ALPHA = 0.9        # Learning rate
GAMMA = 0.5        # Discount factor
EPSILON_START = 0.99  # Initial exploration rate
```

**DQN/DDQN:**
```python
ALPHA = 0.001      # Learning rate
GAMMA = 0.999      # Discount factor  
EPSILON_START = 0.999  # Initial exploration rate
EPSILON_MIN = 0.1   # Minimum exploration rate
EPSILON_DECAY = 0.995  # Exploration decay factor
```

**DDQN Specific:**
```python
UPDATE_FREQUENCY = 100  # Target network sync frequency
```

### Reward Configuration

Customize rewards in each launcher:
```python
R1 = (goal_reward, dragon_penalty, out_of_bounds_penalty, step_penalty)
```

## 🛠️ Development

### Adding New Algorithms

1. Create new launcher class inheriting from `Launcher`
2. Implement `launch_training()` and `launch_test()` methods
3. Add algorithm choice to `main.py` parser
4. Update `view/engine.py` with new launcher instance

### Code Structure

- **Core algorithms**: Located in `core/` directory
- **Utilities**: Shared functions in `core/utils*.py`
- **Visualization**: Game engine and sprites in `view/`
- **Entry points**: `main.py` and `demo_ddqn.py`

### Dependencies

```python
tensorflow>=2.0.0     # Deep learning framework
numpy>=1.19.0         # Numerical computing
pygame>=2.0.0         # Game visualization
tqdm>=4.60.0          # Progress bars
matplotlib>=3.3.0     # Plotting and visualization
```

## 📈 Algorithm Comparison

| Algorithm | State Representation | Training Stability | Memory Usage | Convergence |
|-----------|---------------------|-------------------|--------------|-------------|
| Q-Learning | Discrete (table) | High | High | Good |
| DQN | Continuous (neural) | Medium | Low | Very Good |
| DDQN | Continuous (neural) | High | Low | Excellent |

### When to Use Each Algorithm

- **Q-Learning**: Small state spaces, interpretable Q-values
- **DQN**: Large state spaces, visual input, good baseline performance
- **DDQN**: When DQN shows overestimation bias, requires stable training

## 🧪 Testing

### Unit Testing

Test individual components:
```bash
# Test DDQN import
python3 -c "import core.doubleqdeeplearning; print('DDQN import OK')"

# Test engine initialization
python3 -c "from view.engine import Engine; engine = Engine(4,4); print('Engine OK')"
```

### Performance Testing

Run comparative analysis:
```python
# Compare algorithms on same environment
from core.doubleqdeeplearning import LauncherDDQN
ddqn = LauncherDDQN(4, 4)
ddqn.launch_training()
ddqn.launch_test()
```


**Happy Learning!** 🤖🎯
