# Example file showing a circle moving on screen
import pygame
import random

# Parameters for every instance
pygame.init()
pygame.display.set_caption("PBSnakeAI")
score_font = pygame.font.SysFont("calibri", 20)
block_size = 20
number_of_blocks = 20

class PBSnakeAIgame:
    # Initialization of the game
    def __init__(self) -> None:
        self.screen_size=number_of_blocks*block_size
        self.screen = pygame.display.set_mode([self.screen_size,self.screen_size]
        )
        self.clock = pygame.time.Clock()
        self.head = pygame.Vector2(round(number_of_blocks / 4) * 20,round(number_of_blocks / 4) * 20)
        self.snake = [
            self.head,
            pygame.Vector2(self.head.x - block_size, self.head.y),
        ]
        self.score = 0
        self.food = pygame.Vector2(0,0)
        self.running = True
        self.direction = 0
        self.place_food()

    # Drawing
    def drawing_ui(self) -> None:
        self.screen.fill("black") 
        for snake_part in self.snake:
            pygame.draw.rect(self.screen, "red",[snake_part.x,snake_part.y,block_size,block_size])

        score_value = score_font.render("Score: " + str(self.score), True, "red")
        self.screen.blit(score_value, [0, 0])
        pygame.draw.rect(self.screen, "green", [self.food.x + 5, self.food.y + 5, 10, 10])
        pygame.display.flip()

    # Every game frame
    def game_frame(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    self.direction = 0
                if event.key == pygame.K_LEFT:
                    self.direction = 1
                elif event.key == pygame.K_UP:
                    self.direction = 2
                elif event.key == pygame.K_DOWN:
                    self.direction = 3

        self.move_snake(self.direction)
        self.snake.insert(0, self.head)

        if self.detect_collision():
            running = False
            return running, self.score
        
        if self.head == self.food:
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()

        self.drawing_ui()
        self.clock.tick(30)
        return self.running, self.score

    #Detecting collisions between walls and snake itself
    def detect_collision(self) -> None:
        if (
            self.head.x > self.screen_size - block_size
            or self.head.x < 0
            or self.head.y > self.screen_size - block_size
            or self.head.y < 0
        ):
            return True
        if self.head in self.snake[1:]:
            return True

    #Generating new place for a food
    def place_food(self):
        self.food = pygame.Vector2(
            random.randint(0, number_of_blocks - 1) * 20,
            random.randint(0, number_of_blocks - 1) * 20,
        )
        if self.food in self.snake:
            self.place_food()

    #Moving snake
    def move_snake(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == 0:
            x += block_size
        elif direction == 1:
            x -= block_size
        elif direction == 2:
            y -= block_size
        elif direction == 3:
            y += block_size

        self.head = pygame.Vector2(x,y)

if __name__=="__main__":
    game_object = PBSnakeAIgame()
    while(True):
        running, score = game_object.game_frame()
        if running == 0: break
    print("Wynik =",score)
    pygame.quit()