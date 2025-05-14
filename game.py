import pygame
import random
from enum import Enum
import numpy as np
from typing import Tuple

from config import NUMBER_OF_BLOCKS, GAME_SPEED, COLLISION_PENALTY, FOOD_REWARD, FINISH_REWARD

# Initialize pygame and set up display
pygame.init()
pygame.display.set_caption("PBSnakeAI")
score_font = pygame.font.SysFont("calibri", 20)

class Direction(Enum):
    """Possible movement directions for the snake."""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class PBSnakeAiGame:
    """Main game class for the Snake AI environment."""
    def __init__(self):
        self.block_size = 30
        self.screen_size = NUMBER_OF_BLOCKS * self.block_size
        self.screen = pygame.display.set_mode([self.screen_size, self.screen_size])
        self.clock = pygame.time.Clock()
        self.snake_reset()
        self.boundaries = self.get_boundaries()

    def snake_reset(self):
        """Resets the snake and game state."""
        self.score = 0
        self.ticks = 0
        self.head = pygame.Vector2(
            round(NUMBER_OF_BLOCKS / 4) * self.block_size, round(NUMBER_OF_BLOCKS / 4) * self.block_size
        )
        self.snake = [
            self.head,
            pygame.Vector2(self.head.x - self.block_size, self.head.y),
        ]
        self.direction = Direction.RIGHT
        self._place_food()

    def _place_food(self):
        """Places food at a random location not occupied by the snake."""
        self.food = pygame.Vector2(
            random.randint(1, NUMBER_OF_BLOCKS - 2) * self.block_size,
            random.randint(1, NUMBER_OF_BLOCKS - 2) * self.block_size,
        )
        if self.food in self.snake:
            self._place_food()

    def get_boundaries(self):
        """Returns a list of boundary cells as Vector2 objects."""
        boundaries = []
        for i in range(NUMBER_OF_BLOCKS):
            for j in range(NUMBER_OF_BLOCKS):
                if (i == 0 or j == 0) or (i == NUMBER_OF_BLOCKS - 1 or j == NUMBER_OF_BLOCKS - 1):
                    boundaries.append(pygame.Vector2(i * self.block_size, j * self.block_size))
        return boundaries

    def _move_snake(self, action):
        """
        Moves the snake in the direction based on the action.
        action: [straight, right turn, left turn]
        """
        cw = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = cw.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            self.direction = cw[idx]
        elif np.array_equal(action, [0, 1, 0]):
            self.direction = cw[(idx + 1) % 4]
        elif np.array_equal(action, [0, 0, 1]):
            self.direction = cw[(idx - 1) % 4]

        x, y = self.head.x, self.head.y
        if self.direction == Direction.UP:
            y -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        self.head = pygame.Vector2(x, y)

    def _drawing_ui(self):
        """Draws the game UI including snake, food, boundaries, and score."""
        self.screen.fill(pygame.Color("#100C08"))
        for snake_part in self.snake:
            pygame.draw.rect(self.screen, "green", [snake_part.x, snake_part.y, self.block_size, self.block_size])
        score_value = score_font.render("Score: " + str(self.score), True, "red")
        self.screen.blit(score_value, [0, 0])
        pygame.draw.rect(self.screen, "red", [self.food.x, self.food.y, self.block_size, self.block_size])
        for cell in self.boundaries:
            pygame.draw.rect(self.screen, "gray", [cell.x, cell.y, self.block_size, self.block_size])
        pygame.display.flip()

    def detect_collision(self, pt=None):
        """
        Checks if the given point collides with boundaries or the snake itself.
        If pt is None, checks the snake's head.
        """
        if pt is None:
            pt = self.head
        if pt in self.boundaries or pt in self.snake[1:]:
            return True
        return False

    def game_frame(self, action) -> Tuple[bool, int, int]:
        """
        Executes one frame of the game.
        Returns: (game_over, score, step_reward)
        """
        self.ticks += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self._move_snake(action)
        self.snake.insert(0, self.head)
        step_reward = 0
        game_over = False

        # Check if snake has filled all available spaces (excluding boundaries)
        max_snake_length = (NUMBER_OF_BLOCKS - 2) * (NUMBER_OF_BLOCKS - 2)
        finished = len(self.snake) == max_snake_length

        if self.detect_collision() or self.ticks > 70 * len(self.snake):
            step_reward -= COLLISION_PENALTY
            game_over = True

        if self.head == self.food:
            self.score += 1
            step_reward += FOOD_REWARD
            self._place_food()
        else:
            self.snake.pop()

        # Grant finish reward only if snake filled all spaces
        if finished:
            step_reward += FINISH_REWARD
            game_over = True

        self._drawing_ui()
        self.clock.tick(GAME_SPEED)
        return game_over, self.score, step_reward
