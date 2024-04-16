import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import random
from math import exp
from collections import deque, namedtuple

BATCH_SIZE = 128
MEMORY_SIZE = 10000
GAMMA = 0.9
EPS_START = 0.95
EPS_END = 0.01
EPS_DECAY = 1000
TAU = 0.005
LR = 0.001

INPUT_SIZE = 104
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "game_over")
)


class ReplayMemory:
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
        self.layer1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.layer2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer3 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.actions_taken = 0

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Trainer:
    def __init__(self):
        self.criterion = nn.MSELoss()
        self.steps_done = 0

        self.model = Deep_QNet().to(device)
        self.target_model = Deep_QNet().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(MEMORY_SIZE)
        torch.load("model.pth")

    def get_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * exp(
            -1.0 * self.steps_done / EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float, device=device)
                predicted_move = self.model(state_tensor.unsqueeze(0)).argmax(1)
            move = F.one_hot(predicted_move, num_classes=OUTPUT_SIZE).view(-1)
        else:
            move_idx = random.randint(0, OUTPUT_SIZE - 1)
            move = torch.zeros(OUTPUT_SIZE, device=device)
            move[move_idx] = 1
        return move.tolist()

    def target_network_update(self):
        target_weights = self.target_model.state_dict()
        model_weights = self.model.state_dict()
        for i in model_weights:
            target_weights[i] = model_weights[i] * TAU + target_weights[i] * (1 - TAU)
        self.target_model.load_state_dict(target_weights)

    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = self.memory.sample(BATCH_SIZE)
        states = Transition(*zip(*minibatch))

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

        next_state = torch.zeros(BATCH_SIZE, device=device)
        next_state[non_final_mask] = (self.target_model(non_final_next_states).max(1)[0])

        Q_expected = self.model(state).gather(1, action.argmax(dim=1, keepdim=True))
        Q_target = (next_state * GAMMA) + reward

        loss = self.criterion(Q_expected, Q_target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
