# Example file showing a circle moving on screen
import pygame
import random

def Your_score(score,tick):
    value = score_font.render("Score: " + str(score), True, "red")
    value_2 =score_font.render("Ticks: " + str(tick), True, "red")
    screen.blit(value, [0, 0])
    screen.blit(value_2, [0, 20])

def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(screen, "red", [x[0], x[1], snake_block, snake_block])

# pygame setup
pygame.init()
running = True
dt = 0 #Ticks counter
dir_x=1 #X Direction of the snake
dir_y=0 #Y Direction of the snake
block_size=20; #Size of a single block
number_of_blocks=40; #How manty blocks in X and Y
snake_length=2; #Snake length
snake_List = [] 

screen_size=(number_of_blocks*block_size)

screen = pygame.display.set_mode([screen_size,screen_size])
score_font = pygame.font.SysFont("comicsansms", 20)
clock = pygame.time.Clock()
pygame.display.set_caption('PBSnakeAI')
player_pos = pygame.Vector2(random.randint(1,round(number_of_blocks-1/2))*20, random.randint(1,number_of_blocks-1)*20)
food = pygame.Vector2(random.randint(1,number_of_blocks-1)*20,random.randint(1,number_of_blocks-1)*20)


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        dir_x=0
        dir_y=-1
    if keys[pygame.K_s]:
        dir_x=0
        dir_y=1
    if keys[pygame.K_a]:
        dir_x=-1
        dir_y=0
    if keys[pygame.K_d]:
        dir_x=1
        dir_y=0

    player_pos.x+=block_size*dir_x
    player_pos.y+=block_size*dir_y

    snake_Head = []
    snake_Head.append(player_pos.x)
    snake_Head.append(player_pos.y)
    snake_List.append(snake_Head)

    if(player_pos.x>=screen_size-18 or player_pos.x<=-2 or player_pos.y>=screen_size-18 or player_pos.y<=-2):
        print("koniec")
        running=False
 
    if len(snake_List) > snake_length:
            del snake_List[0]

    our_snake(block_size, snake_List)
    pygame.draw.rect(screen,"green",[food.x+5,food.y+5,10,10])
    
    Your_score(snake_length - 2,dt)

    if player_pos.x == food.x and player_pos.y == food.y:
        snake_length+=1
        food = pygame.Vector2(random.randint(0,number_of_blocks)*20,random.randint(0,number_of_blocks)*20)

    pygame.display.flip()

    screen.fill("black") #Clearing screen
    dt+=1
    clock.tick(block_size)

pygame.quit()