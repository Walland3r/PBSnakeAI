import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import random
from math import exp
from collections import deque,namedtuple

BATCH_SIZE = 128
MEMORY_SIZE = 10000
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 0.001

INPUT_SIZE = 12
HIDDEN_SIZE = 128
OUTPUT_SIZE = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward' , 'next_state', 'game_over'))

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class Deep_QNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(INPUT_SIZE,  HIDDEN_SIZE)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer3 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.actions_taken = 0

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
class Trainer():
    def __init__(self):
        self.criterion = nn.MSELoss()
        self.steps_done = 0

        self.predict_model = Deep_QNet().to(device)
        self.target_model   = Deep_QNet().to(device)
        self.target_model.load_state_dict(self.predict_model.state_dict())
        self.optimizer = optim.Adam(self.predict_model.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
    
    def get_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        move=[0,0,0]
        if sample > eps_threshold:
            predicted_move = self.predict_model(torch.tensor(state, dtype=torch.float))
            move[torch.argmax(predicted_move).item()] = 1
        else:
            move[random.randint(0, 2)] = 1
        return move
    
    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return  # Jeśli pamięć nie zawiera wystarczającej liczby próbek, to nie można optymalizować modelu

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))  # Rozpakowanie krotek Transition w listy stanów, akcji, nagród, itd.

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None], dtype=torch.float, device=device)

        state_batch = torch.tensor(batch.state, dtype=torch.float, device=device)
        action_batch = torch.tensor(batch.action, device=device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float, device=device)

        state_action_values = self.predict_model(state_batch).gather(1, action_batch.argmax(dim=1, keepdim=True))

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.predict_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
