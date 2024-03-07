# Example file showing a circle moving on screen
import pygame
import random

# Onscreen score info
def Your_score(score,tick):
    value = score_font.render("Score: " + str(score), True, "red")
    value_2 =score_font.render("Ticks: " + str(tick), True, "red")
    screen.blit(value, [0, 0])
    screen.blit(value_2, [0, 20])

#Parameters
dt = 0 #Ticks counter
dir_x=1 #X Direction of the snake
dir_y=0 #Y Direction of the snake
block_size=20; #Size of a single block
number_of_blocks=40; #How manty blocks in X and Y
snake_length=3; #Starting snake length

#Initialization
pygame.init()
snake_list = [] #List of snake compartments
screen_size=(number_of_blocks*block_size) #Size of the game window
screen = pygame.display.set_mode([screen_size,screen_size]) #Initialization of the window
score_font = pygame.font.SysFont("comicsansms", 20) #Font for the onscreen score
clock = pygame.time.Clock() #Clock initialization
pygame.display.set_caption('PBSnakeAI') #Window name
player_pos = pygame.Vector2(round(number_of_blocks/4)*20) #Starting player position
food = pygame.Vector2(random.randint(1,number_of_blocks-1)*20,random.randint(1,number_of_blocks-1)*20) #Starting food position
running = True # While argument

#Main game loop
while running:   
                     
    #Clicked buttons handler
    pygame.event.get()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] and dir_y!=1:
        dir_x=0
        dir_y=-1
    if keys[pygame.K_s] and dir_y!=-1:
        dir_x=0
        dir_y=1
    if keys[pygame.K_a] and dir_x!=1:
        dir_x=-1
        dir_y=0
    if keys[pygame.K_d] and dir_x!=-1:
        dir_x=1
        dir_y=0

    #Moving snake head
    player_pos.x+=block_size*dir_x
    player_pos.y+=block_size*dir_y

    #Making list of snake parts
    snake_Head = []
    snake_Head.append(player_pos.x)
    snake_Head.append(player_pos.y)
    snake_list.append(snake_Head)

    #Screen edges handler
    if(player_pos.x>=screen_size-18 or player_pos.x<=-2 or player_pos.y>=screen_size-18 or player_pos.y<=-2):
        print("koniec")
        running=False
 
    #Removing tail
    if len(snake_list) > snake_length:
            del snake_list[0]

    #Snake drawing
    for x in snake_list:
        pygame.draw.rect(screen, "red", [x[0], x[1], block_size, block_size])

    #Scoreboard handling
    Your_score(snake_length - 2,dt)

    #Snake self collision
    for i in snake_list[:-1]:
        if i == snake_Head:
            running=False

    #Food spawn and draw
    if player_pos.x == food.x and player_pos.y == food.y:
        snake_length+=1
        food = pygame.Vector2(random.randint(1,number_of_blocks-1)*20,random.randint(1,number_of_blocks-1)*20)
    pygame.draw.rect(screen,"green",[food.x+5,food.y+5,10,10])

    pygame.display.flip()
    screen.fill("black") #Clearing screen
    dt+=1 #Tick coutner
    clock.tick(30)#Setting tickrate

print("Score:",snake_length-2,"Ticks: ",dt)
pygame.quit()