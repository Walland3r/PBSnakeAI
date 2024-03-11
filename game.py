import pygame
import random
from enum import Enum
import numpy as np
from typing import Tuple

# Parameters for every instance
pygame.init()
pygame.display.set_caption("PBSnakeAI")
score_font = pygame.font.SysFont("calibri", 20)
block_size = 20
number_of_blocks = 10


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class PBSnakeAIgame:
    # Initialization of the game
    def __init__(self) -> None:
        self.screen_size = number_of_blocks * block_size
        self.screen = pygame.display.set_mode([self.screen_size, self.screen_size])
        self.clock = pygame.time.Clock()
        self.ticks = 0
        self.snake_reset()

    # Reset of our snake
    def snake_reset(self) -> None:
        self.score = 0
        self.ticks = 0
        self.reward = 0
        self.game_over = 0

        self.head = pygame.Vector2(
            round(number_of_blocks / 4) * 20, round(number_of_blocks / 4) * 20
        )
        self.snake = [
            self.head,
            pygame.Vector2(self.head.x - block_size, self.head.y),
        ]
        self.direction = Direction.RIGHT
        self._place_food()

    # Generating new place for a food
    def _place_food(self) -> None:
        self.food = pygame.Vector2(
            random.randint(0, number_of_blocks - 1) * 20,
            random.randint(0, number_of_blocks - 1) * 20,
        )
        if self.food in self.snake:
            self._place_food()

    # Moving snake
    def _move_snake(self, action) -> None:
        cw = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = cw.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            self.direction = cw[idx]  # No change
        elif np.array_equal(action, [0, 1, 0]):
            self.direction = cw[(idx + 1) % 4]  # Turn clockwise
        elif np.array_equal(action, [0, 0, 1]):
            self.direction = cw[(idx - 1) % 4]  # Turn counter-clockwise

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.UP:
            y -= block_size
        elif self.direction == Direction.DOWN:
            y += block_size
        elif self.direction == Direction.RIGHT:
            x += block_size
        elif self.direction == Direction.LEFT:
            x -= block_size

        self.head = pygame.Vector2(x, y)

    # Drawing
    def _drawing_ui(self) -> None:
        # Background
        self.screen.fill(pygame.Color("#100C08"))
        # Snake
        for snake_part in self.snake:
            pygame.draw.rect(
                self.screen,
                "green",
                [snake_part.x, snake_part.y, block_size, block_size],
            )
        pygame.draw.circle(self.screen, "black", [self.head.x + 15, self.head.y + 5], 4)
        pygame.draw.circle(
            self.screen, "black", [self.head.x + 15, self.head.y + 15], 4
        )
        # Score
        score_value = score_font.render("Score: " + str(self.score), True, "red")
        self.screen.blit(score_value, [0, 0])
        # Food
        pygame.draw.rect(self.screen, "red", [self.food.x + 5, self.food.y + 5, 10, 10])
        pygame.display.flip()
        while(True):
            pass

    # Detecting collisions between walls and snake itself
    def detect_collision(self, pt=None) -> bool:
        if pt == None:
            pt = self.head
        # Hit the wall
        if (
            pt.x > self.screen_size - block_size
            or pt.x < 0
            or pt.y > self.screen_size - block_size
            or pt.y < 0
        ):
            return pt
        # Hit itself
        if pt in self.snake[1:]:
            return True

    # Every game frame
    def game_frame(self, action) -> Tuple[bool, int, int]:
        reward = 0
        game_over = False
        self.ticks += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self._move_snake(action)
        self.snake.insert(0, self.head)

        if self.detect_collision() or self.ticks > 100 * len(self.snake):
            reward -= 10
            game_over = True
            return game_over, self.score, reward

        if self.head == self.food:
            self.score += 1
            self.reward += 10
            self._place_food()
        else:
            self.snake.pop()

        self._drawing_ui()
        self.clock.tick(30)
        return self.game_over, self.score, reward

    def snake_vision(self):
        up=-1
        down=-2
        left=-1
        right=-2
        for i in range(int(self.head.y), -20,-20):
            up+=1
            if self.detect_collision(pygame.Vector2(self.snake[0].x, i)):
                break
        for i in range(int(self.head.y), self.screen_size+20,20):
            down+=1
            if self.detect_collision(pygame.Vector2(self.snake[0].x, i)):
                break
        for i in range(int(self.head.x),-20,-20):
            left+=1
            if self.detect_collision(pygame.Vector2(i, self.snake[0].y)):
                break
        for i in range(int(self.head.x), self.screen_size+20,20):
            right+=1
            if self.detect_collision(pygame.Vector2(i, self.snake[0].y)):
                break
        return up,down,left,right

# Test of the game
def test_game() -> None:
    game = PBSnakeAIgame()
    while True:
        game_over, score, reward = game.game_frame([1, 0, 0])
        if game_over == True:
            game.snake_reset()
    print("Final Score", score)
