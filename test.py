from agent import GameAgent
from game import PBSnakeAiGame
from plotter import plot

agent = GameAgent()
game = PBSnakeAiGame()
scores_list=[]
avg_scores=[]
game_counter=0
score_sum=0

for i in range (100):
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
            game_counter+=1
            break
    scores_list.append(total_score)
    score_sum += total_score
    avg_scores.append(score_sum/game_counter)
    plot(scores_list,avg_scores)
    print(avg_scores[-1])
    game.snake_reset()
