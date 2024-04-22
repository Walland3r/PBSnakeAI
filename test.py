from agent import GameAgent
from game import PBSnakeAiGame

agent = GameAgent()
game = PBSnakeAiGame()

while True:
    current_state = agent.get_gamestate(game)
    predicted_move = agent.trainer.get_action(current_state)
    game_over, total_score, reward = game.game_frame(predicted_move)

    if game_over:
        print(
            "Game:",
            agent.number_of_games,
            "Score:",
            total_score,
            "Reward:",
            reward,
        )
        break
