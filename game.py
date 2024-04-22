import pygame
import random
from enum import Enum
import numpy as np
from typing import Tuple

pygame.init()
pygame.display.set_caption("PBSnakeAI")
score_font = pygame.font.SysFont("calibri", 20)
number_of_blocks = 10
GAME_SPEED = 60


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class PBSnakeAiGame:
    # Initialization of the game
    def __init__(self) -> None:
        self.block_size = 30
        self.screen_size = number_of_blocks * self.block_size
        self.screen = pygame.display.set_mode([self.screen_size, self.screen_size])
        self.clock = pygame.time.Clock()
        self.ticks = 0
        self.reward = 0
        self.snake_reset()
        self.boundaries = self.get_boundaries()

    # Reset whole game
    def snake_reset(self) -> None:
        self.score = 0
        self.ticks = 0
        self.game_over = 0
        self.reward = 0

        self.head = pygame.Vector2(
            round(number_of_blocks / 4) * self.block_size, round(number_of_blocks / 4) * self.block_size
        )
        self.snake = [
            self.head,
            pygame.Vector2(self.head.x - self.block_size, self.head.y),
        ]
        self.direction = Direction.RIGHT
        self._place_food()

    # Generating new place for a food
    def _place_food(self) -> None:
        self.food = pygame.Vector2(
            random.randint(1, number_of_blocks - 2) * self.block_size,
            random.randint(1, number_of_blocks - 2) * self.block_size,
        )
        if self.food in self.snake:
            self._place_food()

    def get_boundaries(self) -> list:
        boundaries = []
        for i in range(number_of_blocks):
            for j in range(number_of_blocks):
                if (i == 0 or j == 0) or (i == number_of_blocks - 1 or j == number_of_blocks - 1):
                    boundaries.append(pygame.Vector2(i*self.block_size, j*self.block_size))
        return boundaries

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
            y -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size

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
                [snake_part.x, snake_part.y, self.block_size, self.block_size],
            )

        # Score
        score_value = score_font.render("Score: " + str(self.score), True, "red")
        self.screen.blit(score_value, [0, 0])

        # Food
        pygame.draw.rect(self.screen,
                         "red",
                         [self.food.x, self.food.y, self.block_size, self.block_size])

        # Boundaries
        for cell in self.boundaries:
            pygame.draw.rect(self.screen,
                             "gray",
                             [cell.x, cell.y, self.block_size, self.block_size])

        pygame.display.flip()

    # Detecting collisions between walls and snake itself
    def detect_collision(self, pt=None) -> bool:
        if pt is None:
            pt = self.head
        # Hit the boundary
        if pt in self.boundaries:
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
            self.reward -= 2
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
