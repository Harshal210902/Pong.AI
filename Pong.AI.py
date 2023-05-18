import pygame
import numpy as np
import random
BLACK = (0,0,0)
WHITE = (255,255,255)
GREEN = (46, 179, 126)
BLUE = (41, 80, 242)
FILL = WHITE
TEXT = BLACK

pygame.init()
layer_structure = [4, 3]

size = (800,600)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("pong")

class PaddleAi:
    xspeed = 0
    
    def __init__(self, x = 450):
        self.x = x
        
    def update(self, x, xspeed):
        self.xspeed = xspeed
        if x>self.x+50 and xspeed>0:
            self.x= self.x+5
        elif x<self.x+50 and xspeed<0:
            self.x= self.x-5
        if self.x > 900:
            self.x = 900
        elif self.x < 0:
            self.x = 0
        
        self.xlast = self.x
    
    def draw(self):
        if self.xspeed > 0:
            pygame.draw.rect(screen,BLACK,[self.x,0,100,20])
            pygame.draw.rect(screen,BLUE,[self.x,1,100-2,20-2])
        elif self.xspeed < 0:
            pygame.draw.rect(screen,BLACK,[self.x,0,100,20])
            pygame.draw.rect(screen,BLUE,[self.x+2,1,100-2,20-2])
        else:
            pygame.draw.rect(screen,BLACK,[self.x,0,100,20])
            pygame.draw.rect(screen,BLUE,[self.x+2,1,100-2,20-2])
            
        
            


class Paddle:
    
    def __init__(self, x = 500, xspeed = 0, coefs = 0, intercepts = 0):
        self.x = x
        self.xlast = x-xspeed
        self.xspeed = xspeed
        self.alive = True
        self.score = 0
        self.command = 2
        self.winner = False
        if coefs == 0:
            self.coefs = self.generateCoefs(layer_structure)
        else:
            self.coefs = coefs
        if intercepts == 0:
            self.intercepts = self.generateIntercepts(layer_structure)
        else:
            self.intercepts = intercepts
     
    def generateCoefs(self, layer_structure):
        coefs = []
        for i in range(len(layer_structure)-1):
            coefs.append(np.random.rand(layer_structure[i], layer_structure[i+1])*2-1)
        return coefs
       
    def generateIntercepts(self, layer_structure):
        intercepts = []
        for i in range(len(layer_structure)-1):
            intercepts.append(np.random.rand(layer_structure[i+1])*2-1)
        return intercepts
    
    def mutateWeights(self):
        newWeights = self.weights.copy()
        for i in range(len(newWeights)):
            for row in range(len(newWeights[i])):
                for col in range(len(newWeights[i][row])):
                    newWeights[i][row][col] = np.random.normal(newWeights[i][row][col], 1)
        return newWeights
     
    def mutateBiases(self):
        newBiases = self.biases.copy()
        for i in range(len(newBiases)):
            for row in range(len(newBiases[i])):
                newBiases[i][row] = np.random.normal(newBiases[i][row], 1)
        return newBiases
    
    def mutate(self):
        return Paddle(coefs = self.mutateCoefs(), intercepts = self.mutateIntercepts())
        
    def reset(self):
        self.x = 500
        self.xlast = 500
        self.xspeed = 0
        self.alive = True
        self.score = 0
       
    def update(self):
        self.xlast = self.x
        self.x += self.xspeed
        if self.x < 0:
            self.x = 0
        elif self.x > size[0]-100:
            self.x=size[0]-100
        
        self.xlast = self.x
       
    def draw(self):
        if self.winner == False:
            pygame.draw.rect(screen,BLACK,[self.x,size[1]-20,100,20])
            pygame.draw.rect(screen,GREEN,[self.x+2,size[1]-18,100-4,20-4])
        else:
            pygame.draw.rect(screen,BLACK,[self.x,size[1]-20,100,20])
            pygame.draw.rect(screen,BLUE,[self.x+2,size[1]-18,100-4,20-4])

class Ball:
    
    def __init__(self, x, xspeed):
        self.x = x
        self.xspeed = xspeed
        self.y = 500
        self.yspeed = 5
        self.xlast = self.x-self.xspeed
        self.ylast = self.y-self.yspeed
        self.alive = True
    
    def update(self, paddle):
        self.xlast = self.x
        self.ylast = self.y
        
        self.x += self.xspeed
        self.y += self.yspeed
        
        if self.x<0:
            self.x=0
            self.xspeed = self.xspeed * -1
            
        elif self.x>size[0]-15:
            self.x=size[0]-15
            self.xspeed = self.xspeed * -1
        
        elif self.x>paddle.x and self.x<paddle.x+100 and self.ylast<size[1]-35 and self.y>=size[1]-35:
            self.yspeed = self.yspeed * -1
            paddle.score = paddle.score + 1
            
        elif self.y<=15:
            self.yspeed = self.yspeed * -1
            
        elif self.y>size[1]:
            self.yspeed = self.yspeed * -1
            paddle.alive = False
            paddle.score -= round(abs((paddle.x+50)-self.x)/100,2)
            
    def draw(self):
        pygame.draw.rect(screen,BLACK,[self.x,self.y,15,15])
    
def calculateOutput(input, layer_structure, coefs, intercepts, g="identity"):
    layers = [np.transpose(input)]
    previousLayer = np.transpose(input)
    
    GREENuced_layer_structure = layer_structure[1:]
    for k in range(len(GREENuced_layer_structure)):
        currentLayer = np.empty((GREENuced_layer_structure[k],1))
        result = np.matmul(np.transpose(coefs[k]),previousLayer) + np.transpose(np.array([intercepts[k]]))
        for i in range(len(currentLayer)):
            if g == "identity":
                currentLayer[i] = result[i]
            else:
                currentLayer[i] = max(0, result[i])
            layers.append(currentLayer)
        previousLayer = currentLayer.copy()
    return(layers[-1].tolist().index(max(layers[-1].tolist())))    
    
def mutateCoefs(coefs):
    newCoefs = []
    for array in coefs:
        newCoefs.append(np.copy(array))
    for i in range(len(newCoefs)):
        for row in range(len(newCoefs[i])):
            for col in range(len(newCoefs[i][row])):
                newCoefs[i][row][col] = np.random.normal(newCoefs[i][row][col], 1)
    return newCoefs

def mutateIntercepts(intercepts):
    newIntercepts = []
    for array in intercepts:
        newIntercepts.append(np.copy(array))
    for i in range(len(newIntercepts)):
        for row in range(len(newIntercepts[i])):
            newIntercepts[i][row] = np.random.normal(newIntercepts[i][row], 1)
    return newIntercepts 
    

done = False
score = 0
command = "stop"
clock=pygame.time.Clock()

COUNT = 100

sca_pong = random.randint(0,1000)
sca_pong_xspeed = ((-1)**(random.randint(1,2)))*5
paddleAi = PaddleAi(sca_pong)
paddles = []
balls = []
for i in range(100):
    paddles.append(Paddle())
    balls.append(Ball(sca_pong, sca_pong_xspeed))

winner = paddles[-1]
paddles[-1].winner = True

#game's main loop  
generation = 1
while not done:
    screen.fill(FILL)
    
    still_alive = 0
    high_score = -9e99
    high_score_index = -1
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    
    for i, paddle in enumerate(paddles):
        input = np.array([[paddle.x, balls[i].x, balls[i].y, balls[i].xspeed]])
        paddle.command = calculateOutput(input, layer_structure, paddle.coefs, paddle.intercepts)
        
        if paddle.command == 0:
                paddle.xspeed = -5
        elif paddle.command == 1:
                paddle.xspeed = 5
        elif paddle.command == 2:
                paddle.xspeed = 0
                
        paddleAi.draw()
        
        if paddle.alive == True:
            paddle.update()  
            balls[i].update(paddle)
            paddleAi.update(balls[i].x, balls[i].xspeed)
            still_alive += 1

        if paddle.score > high_score:
            high_score = paddle.score
            high_score_index = i
            winner = paddles[i]
            winner.winner = True
            
        if paddle.alive and paddle != winner:
            paddle.draw()
            balls[i].draw()
            paddle.winner = False
    
    paddles[high_score_index].draw()
    balls[high_score_index].draw()
        
    if still_alive == 0:
        generation += 1
        winner.reset()
        print(high_score_index)
        paddles = []
        balls = []
        current_x = random.randint(0,1000)
        current_xspeed = ((-1)**(random.randint(1,2)))*5
        for i in range(COUNT-1):
            paddles.append(Paddle(coefs = mutateCoefs(winner.coefs), intercepts = mutateIntercepts(winner.intercepts)))
            balls.append(Ball(current_x,current_xspeed))
        paddles.append(winner)
        balls.append(Ball(current_x, current_xspeed))

    
    
  
    pygame.display.flip()         
    clock.tick(60)
    
pygame.quit()
