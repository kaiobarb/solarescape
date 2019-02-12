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
        self.mass = mass
        self.vx = float(velocityx)
        self.vy = float(velocityy)
        if not diameter: 
            self.size = (self.mass/pi*5)**(1/2.0)
        else:
            self.size = diameter
        self.color = (int(255-random.random()*200),int(255-random.random()*200),int(255-random.random()*200))

    def render(self):
        pygame.draw.circle(self.screen, self.color, [self.x, self.y], self.size)

class App(object):

    def __init__(self, width=600, height=600):
        pygame.init()
        pygame.display.set_caption("Press ESC to quit")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.font = pygame.font.SysFont('mono', 20, bold=True)
        self.bodies = []
        # maintains information between bodies to ease calculations
        self.values = {}

    def interact(self, bodyA, bodyB):
        # calculate force, distance between A and B
        # Use those values to update acceleration, and distance btwn

        # acceleration = force / mass
        # (<x, y>)velocity = acceleration * (<x, y>)Distance/TotalDistance

        # Save values[(A, B)] = (force, (<x, y>)distance, TotalDistance) 
        pass
        
    def move(self):
        # call interact on all bodies
        self.values = {}
        for cA in self.bodies:
            for cB in self.bodies:
                if cA != cB:
                    self.interact(cA,cB)
        # update positions of all bodies
        for obj in self.bodies:
            obj.x += obj.vx
            obj.y += obj.vy

    def draw(self):
        # Call each body's render function here
        pass

    def run(self):
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
        pygame.quit()


if __name__ == '__main__':

    # call with width of window
    App(640, 400).run()
