import numpy as np
import torch
import random
from collections import deque
from game import PBSnakeAIgame
from game import Direction
from model import Linear_QNet, QLineTrainer
from helper import plot

class GameAgent:
    def __init__(self) -> None:
        self.memory = deque(maxlen=200000)
        self.epsilon = 0
        self.gamma = 0.9
        self.number_of_games = 0

        self.model = Linear_QNet(12, 256, 3)
        self.trainer = QLineTrainer(self.model, learning_rate=0.001, gamma=self.gamma)

    def get_gamestate(self, game):
        # Current snake direction
        going_up = game.direction == Direction.UP
        going_down = game.direction == Direction.DOWN
        going_right = game.direction == Direction.RIGHT
        going_left = game.direction == Direction.LEFT

        danger_up, danger_down, danger_left, danger_right = game.snake_vision()
        state = [
            going_up,
            going_down,
            going_right,
            going_left,

            danger_up / 10,
            danger_down / 10,
            danger_left / 10,
            danger_right / 10,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]
        return np.array(state, dtype=int)

    def remember(self, current_state, action, reward, next_state, game_over):
        self.memory.append((current_state, action, reward, next_state, game_over))

    def train_long(self):
        if len(self.memory) > 1000:
            sample = random.sample(self.memory, 1000)
        else:
            sample = self.memory
        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.step(states, actions, rewards, next_states, dones)

    def train_short(self, current_state, action, reward, next_state, game_over):
        self.trainer.step(current_state, action, reward, next_state, game_over)

    def get_action(self, state):
        self.epsilon = 80 - self.number_of_games
        move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move[random.randint(0, 2)] = 1
        else:
            predicted_move = self.model(torch.tensor(state, dtype=torch.float))
            move[torch.argmax(predicted_move).item()] = 1
        return move


def train():
    scores_list = []
    avg_scores = []
    total_score = 0
    score_sum = 0
    best_score = 0
    agent = GameAgent()
    game = PBSnakeAIgame()

    while True:
        current_state = agent.get_gamestate(game)
        predicted_move = agent.get_action(current_state)
        game_over, total_score, reward = game.game_frame(predicted_move)
        next_state = agent.get_gamestate(game)

        agent.train_short(current_state, predicted_move, reward, next_state, game_over)
        agent.remember(current_state, predicted_move, reward, next_state, game_over)

        if game_over:
            agent.train_long()
            agent.number_of_games += 1
            game.snake_reset()

            if total_score > best_score:
                best_score = total_score
                agent.model.save()
            print(
                "Game:",
                agent.number_of_games,
                "Score:",
                total_score,
                "Record:",
                best_score,
                "Reword:",
                reward,
            )
            scores_list.append(total_score)
            score_sum += total_score
            mean_score = total_score / agent.number_of_games
            avg_scores.append(mean_score)
            plot(scores_list, avg_scores)

if __name__ == "__main__":
    train()
