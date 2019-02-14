import pygame
import calculations
from pygame.locals import *
import sys , time , random
from math import *
from decimal import *
 

class Body:
    def __init__(self, screen, startx, starty, velocityx, velocityy, mass, diameter):
        self.screen = screen
        self.x = startx
        self.y = starty
        self.m = mass
        self.vx = float(velocityx)
        self.vy = float(velocityy)
        if not diameter: 
            self.size = (self.m/pi*5)**(1/2.0)
        else:
            self.size = diameter
        self.color = (int(255-random.random()*200),int(255-random.random()*200),int(255-random.random()*200))

    def render(self):
        pygame.draw.circle(self.screen, self.color, [int(self.x), int(self.y)], self.size)

class App(object):
    def __init__(self, width=900, height=900):
        pygame.init()
        pygame.display.set_caption("Press ESC to quit")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
        self.background = pygame.Surface(self.screen.get_size())
        self.background.convert() 
        self.font = pygame.font.SysFont('mono', 20, bold=True)
        self.bodies = []
        # maintains information between bodies to ease calculations
        self.values = {}

    def interact(self, A, B):
        # calculate force, distance between A and B
        dx = B.x - A.x
        dy = B.y - A.y
        force = calculations.force(dx, dy, A.m, A.size, B.m, B.size)
        dist = calculations.dist(dx, dy, A.size, B.size)
        # Use those values to update acceleration, and distance btwn
        accelerationA = force / (A.m * 10000000000)
        accelerationB = force / (B.m * 10000000000)
        compAx = dx / dist
        compAy = dy / dist
        compBx = -dx / dist
        compBy = -dy / dist
        A.vx += accelerationA * compAx
        A.vy += accelerationA * compAy
        B.vx += accelerationB * compBx
        B.vy += accelerationB * compBy

        # A.x -= A.vx
        # A.y -= A.vy
        # B.x -= B.vx
        # B.y -= B.vy
        
        
    def move(self):
        # call interact on all bodies
        numBodies = range(len(self.bodies))
        for i in numBodies:
            for j in numBodies[i+1:]:
                    #print(i, self.bodies[i])
                    self.interact(self.bodies[i], self.bodies[j])
        for body in self.bodies:
            body.x += body.vx
            body.y += body.vy

    def draw(self):
        # Call each body's render function here
        for body in self.bodies:
            body.render()
        pass


    def run(self):
        self.bodies.append(Body(self.screen, 450, 450, 0, 0, 10000, 5))
        self.bodies.append(Body(self.screen, 300, 300, 0.05, 0, 100, 2))
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            pygame.display.flip()
            self.screen.blit(self.background, (0, 0))
            #print(self.bodies[0].x)
            self.move()
            self.draw()
        pygame.quit()


if __name__ == '__main__':

    # call with width of window
    App(900, 900).run()
