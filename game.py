import pygame
import random
from enum import Enum
import numpy as np
from typing import Tuple


pygame.init()
pygame.display.set_caption("PBSnakeAI")
score_font = pygame.font.SysFont("calibri", 20)
block_size = 20
number_of_blocks = 10
GAME_SPEED = 0

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
        self.reward = 0
        self.snake_reset()


    # Reset of our snake
    def snake_reset(self) -> None:
        self.score = 0
        self.ticks = 0
        self.game_over = 0
        self.reward = 0

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
        pygame.draw.rect(self.screen, "red", [self.food.x, self.food.y, block_size, block_size])
        pygame.display.flip()

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
            return True
        # Hit itself
        if pt in self.snake[1:]:
            return True
        return False

    # Every game frame
    def game_frame(self, action) -> Tuple[bool, int, int]:
        game_over = False
        self.ticks += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self._move_snake(action)
        self.snake.insert(0, self.head)

        if self.detect_collision() or self.ticks > 70 * len(self.snake):
            self.reward -= 20
            game_over = True

        if self.head == self.food:
            self.score += 1
            self.reward += 1
            self._place_food()
        else:
            self.snake.pop()

        self._drawing_ui()
        self.clock.tick(GAME_SPEED)
        return game_over, self.score, self.reward