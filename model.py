import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from math import exp
from collections import deque, namedtuple
import os.path as path

from config import (
    BATCH_SIZE,
    MEMORY_SIZE,
    GAMMA,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    TAU,
    LR,
    INPUT_SIZE,
    HIDDEN_SIZE,
    OUTPUT_SIZE,
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transition tuple for replay memory
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "game_over")
)

class ReplayMemory:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DeepQNet(nn.Module):
    """Deep Q-Network with three hidden layers."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        )

    def forward(self, x):
        return self.layers(x)

class Trainer:
    """Handles training, action selection, and model persistence."""
    def __init__(self):
        self.criterion = nn.MSELoss()
        self.steps_done = 0

        # Load model if exists, else initialize new
        self.model = DeepQNet().to(device)
        if path.isfile("model.pth"):
            self.model.load_state_dict(torch.load('model.pth', map_location=device))

        self.target_model = DeepQNet().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(MEMORY_SIZE)

    def get_action(self, state):
        """
        Selects an action using epsilon-greedy policy.
        Returns a one-hot encoded action.
        """
        eps_threshold = EPS_END + (EPS_START - EPS_END) * exp(-1.0 * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float, device=device)
                predicted_move = self.model(state_tensor.unsqueeze(0)).argmax(1)
            move = F.one_hot(predicted_move, num_classes=OUTPUT_SIZE).view(-1)
        else:
            move = torch.zeros(OUTPUT_SIZE, device=device)
            move[random.randint(0, OUTPUT_SIZE - 1)] = 1
        return move.tolist()

    def target_network_update(self):
        """
        Soft update of the target network's weights.
        """
        target_weights = self.target_model.state_dict()
        model_weights = self.model.state_dict()
        for key in model_weights:
            target_weights[key] = model_weights[key] * TAU + target_weights[key] * (1 - TAU)
        self.target_model.load_state_dict(target_weights)

    def optimize(self):
        """
        Performs a single optimization step on the model using a minibatch from memory.
        """
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = self.memory.sample(BATCH_SIZE)
        states = Transition(*zip(*minibatch))

        # Prepare tensors
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, states.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.tensor(
            [s for s in states.next_state if s is not None],
            dtype=torch.float,
            device=device,
        )
        state = torch.tensor(states.state, dtype=torch.float, device=device)
        action = torch.tensor(states.action, device=device)
        reward = torch.tensor(states.reward, dtype=torch.float, device=device)

        # Compute Q values
        next_state = torch.zeros(BATCH_SIZE, device=device)
        if non_final_next_states.size(0) > 0:
            next_state[non_final_mask] = self.target_model(non_final_next_states).max(1)[0]
        q_expected = self.model(state).gather(1, action.argmax(dim=1, keepdim=True))
        q_target = (next_state * GAMMA) + reward

        # Compute loss and optimize
        loss = self.criterion(q_expected, q_target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
