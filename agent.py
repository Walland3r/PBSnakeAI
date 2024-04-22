import numpy as np
from game import PBSnakeAiGame
from game import Direction
from model import Trainer
from plotter import plot
from pygame import Vector2
from torch import save


class GameAgent:
    def __init__(self) -> None:
        self.number_of_games = 0
        self.trainer = Trainer()

    def get_gamestate(self, game) -> np.array:
        # Current snake direction
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

            going_up,
            going_down,
            going_right,
            going_left,

            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y
        ] 
        return np.array(state, dtype=bool)

    def remember(self, current_state, action, reward, next_state, game_over):
        self.trainer.memory.push(current_state, action, reward, next_state, game_over)


def train():
    scores_list = []
    avg_scores = []
    total_score = 0
    score_sum = 0
    best_score = 0
    agent = GameAgent()
    game = PBSnakeAiGame()

    while True:
        current_state = agent.get_gamestate(game)
        predicted_move = agent.trainer.get_action(current_state)
        game_over, total_score, reward = game.game_frame(predicted_move)
        next_state = agent.get_gamestate(game)

        agent.remember(current_state, predicted_move, reward, next_state, game_over)
        agent.trainer.optimize()
        agent.trainer.target_network_update()

        if game_over:
            game.snake_reset()
            agent.number_of_games += 1
            if total_score > best_score:
                best_score = total_score
                agent.trainer.target_network_update()
                save(agent.trainer.model, 'model.pth')
            print(
                "Game:",
                agent.number_of_games,
                "Score:",
                total_score,
                "Record:",
                best_score,
                "Reward:",
                reward,
            )
            scores_list.append(total_score)
            score_sum += total_score
            mean_score = score_sum / agent.number_of_games
            avg_scores.append(mean_score)
            plot(scores_list, avg_scores)


if __name__ == "__main__":
    train()
