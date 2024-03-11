import numpy as np
import torch
import random
from collections import deque
from Game import PBSnakeAIgame
from Game import Direction
from matplotlib import pyplot

"""
~~~~ Arguments for our neural network ~~~~
Snake will "look" in 8 directions for food, walls, and itself.
Arguments:
1. Snake moving direction (4), one of them will be '1' at a time.
2. Distance to danger (4).
3. 
Total: 16

"""

class GameAgent:
    def __init__(self) -> None:
        memory = deque(maxlen=200000)
        self.epsilon = 0
        self.gamma = 0
        self.number_of_games = 0

    def get_gamestate(self, game):
        snake_head = game.snake[0]
        window_size = game.screen_size

        # Current snake direction
        going_up = game.direction == Direction.UP
        going_down = game.direction == Direction.DOWN
        going_right = game.direction == Direction.RIGHT
        going_left = game.direction == Direction.LEFT

        danger_up, danger_down, danger_left, danger_right = game.snake_vision()
        print(danger_up,danger_down,danger_left,danger_right)
        state = [
            going_up,
            going_down,
            going_right,
            going_left,

            danger_up,
            danger_down,
            danger_left,
            danger_right
        ]


    def remember(self, current_state, action, reward, next_state):
        pass

    def train_long(self):
        pass

    def train_short(self, current_state, action, reward, next_state):
        pass

    def get_action(self, state):
        pass


def train():
    scores_list = []
    avg_scores = []
    total_score = 0
    best_score = 0
    agent = GameAgent()
    game = PBSnakeAIgame()

    # while True:
    current_state = agent.get_gamestate(game)
    predicted_move = agent.get_action(current_state)
    game_over, total_score, reward = game.game_frame(predicted_move)
    next_state = agent.get_gamestate(game)

    agent.train_short(current_state, predicted_move, reward, next_state)
    agent.remember(current_state, predicted_move, reward, next_state)

    if total_score > best_score:
        best_score = total_score

    if game_over:
        agent.train_long()
        agent.number_of_games += 1
        game.snake_reset()


if __name__ == "__main__":
    train()
