import numpy as np
from pygame import Vector2
from torch import save

from game import PBSnakeAiGame, Direction
from model import Trainer
from plotter import plot

class GameAgent:
    """Agent that interacts with the Snake game and manages memory."""
    def __init__(self):
        self.number_of_games = 0
        self.trainer = Trainer()

    def get_gamestate(self, game):
        """
        Returns the current state as a boolean numpy array.
        State includes danger, direction, and food location.
        """
        going_up = game.direction == Direction.UP
        going_down = game.direction == Direction.DOWN
        going_right = game.direction == Direction.RIGHT
        going_left = game.direction == Direction.LEFT

        point_l = Vector2(game.head.x - game.block_size, game.head.y)
        point_r = Vector2(game.head.x + game.block_size, game.head.y)
        point_u = Vector2(game.head.x, game.head.y - game.block_size)
        point_d = Vector2(game.head.x, game.head.y + game.block_size)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.detect_collision(point_r)) or
            (dir_l and game.detect_collision(point_l)) or
            (dir_u and game.detect_collision(point_u)) or
            (dir_d and game.detect_collision(point_d)),
            # Danger right
            (dir_u and game.detect_collision(point_r)) or
            (dir_d and game.detect_collision(point_l)) or
            (dir_l and game.detect_collision(point_u)) or
            (dir_r and game.detect_collision(point_d)),
            # Danger left
            (dir_d and game.detect_collision(point_r)) or
            (dir_u and game.detect_collision(point_l)) or
            (dir_r and game.detect_collision(point_u)) or
            (dir_l and game.detect_collision(point_d)),
            # Move direction
            going_up,
            going_down,
            going_right,
            going_left,
            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=bool)

    def remember(self, state, action, reward, next_state, done):
        """Stores the experience in replay memory."""
        self.trainer.memory.push(state, action, reward, next_state, done)

def train():
    """
    Main training loop for the agent.
    Handles game interaction, memory, optimization, and plotting.
    """
    scores, avg_scores = [], []
    best_score = 0
    agent = GameAgent()
    game = PBSnakeAiGame()
    total_score = 0

    while True:
        state = agent.get_gamestate(game)
        action = agent.trainer.get_action(state)
        done, score, reward = game.game_frame(action)
        next_state = agent.get_gamestate(game)
        agent.remember(state, action, reward, next_state, done)
        agent.trainer.optimize()
        agent.trainer.target_network_update()

        if done:
            game.snake_reset()
            agent.number_of_games += 1
            if score > best_score:
                best_score = score
                save(agent.trainer.model.state_dict(), 'model.pth')
            scores.append(score)
            total_score += score
            avg_scores.append(total_score / agent.number_of_games)
            print(f"Game: {agent.number_of_games} Score: {score} Record: {best_score}")
            plot(scores, avg_scores)

if __name__ == "__main__":
    train()
