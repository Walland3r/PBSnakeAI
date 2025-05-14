# =========================
# Game Configuration
# =========================

# Number of blocks per row/column (playfield is square)
NUMBER_OF_BLOCKS = 9

# Game speed (frames per second)
GAME_SPEED = 30

# Rewards and penalties
COLLISION_PENALTY = 2      # Penalty for collision (game over)
FOOD_REWARD = 1            # Reward for eating food
FINISH_REWARD = 10         # Reward for filling the board

# =========================
# Model (DQN) Configuration
# =========================

BATCH_SIZE = 128           # Batch size for training
MEMORY_SIZE = 10000        # Replay memory size
GAMMA = 0.9                # Discount factor for future rewards
EPS_START = 0.0           # Initial epsilon for exploration
EPS_END = 0.01             # Final epsilon for exploration
EPS_DECAY = 1000           # Epsilon decay rate
TAU = 0.005                # Soft update parameter for target network
LR = 0.001                 # Learning rate

# Neural Network Architecture
INPUT_SIZE = 11            # Number of input features
HIDDEN_SIZE = 64           # Hidden layer size
OUTPUT_SIZE = 3            # Number of possible actions
