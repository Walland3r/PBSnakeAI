import numpy as np
from game import PBSnakeAIgame
from game import Direction
from model import Trainer
from plotter import plot


class GameAgent:
    def __init__(self) -> None:
        self.number_of_games = 0
        self.trainer = Trainer()

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
        self.trainer.memory.push(current_state,action,reward,next_state,game_over)


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
        predicted_move = agent.trainer.get_action(current_state)
        game_over, total_score, reward = game.game_frame(predicted_move)
        next_state = agent.get_gamestate(game)

        agent.remember(current_state,predicted_move,reward,next_state,game_over)
        agent.trainer.optimize()

        if game_over:
            game.snake_reset()
            agent.number_of_games += 1
            if total_score > best_score:
                best_score = total_score
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
            mean_score = total_score / agent.number_of_games
            avg_scores.append(mean_score)
            plot(scores_list, avg_scores)


if __name__ == "__main__":
    train()
