import pygame
import calculations
import sys , time , random
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from math import *
from pygame.locals import *
from decimal import *

""" 
##########
# Task 1 #
##########

Solar Escape Environment
------------------------

There is always at least one celestial body, the sun, and it is always in the center of the screen (x=width/2, y=height/2)

(Reward values are yet to be finalized)
Reward for entering orbit zone is +10.
Reward for staying within orbit zone and moving forward (around the celestial body) is +20.
Reward for completing orbit is +100.
Reward +1*|current_velocity| every frame. (Thruster adds 0.1 velocity every frame while activated)
Using Thruster is -0.5 reward each frame.

Episode finishes if agent collides (NI) or fails to stay within orbit distance of target planet once orbit radius has been entered.
Fuel is infinite.

(NI: Not Implemented)
Task 1: Orbit once around single target planet (NI)
Task 2: Orbit as many times as possible around  target planet (NI)
Task 3: Orbit first target planet, then orbit second target planet at least once. (NI)
Task 4: Enter orbit radius of every planet, but do not orbit, then exit screen

"""
FPS = 50

THRUSTER_POWER = 0.1



class Body:
    def __init__(self, screen, startx, starty, velocityx, velocityy, mass, diameter):
        self.screen = screen
        self.x = startx
        self.y = starty
        self.m = mass
        self.vx = float(velocityx)
        self.vy = float(velocityy)
        self.target = False
        if not diameter: 
            self.size = ((self.m/pi*5)**(1/2.0))/10
        else:
            self.size = diameter
        self.color = (int(255-random.random()*200),int(255-random.random()*200),int(255-random.random()*200))

    def render(self):
        if self.target:
            s = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
            #s.fill((255,250,205, 128))
            self.screen.blit(s, (self.x - self.size*2//2, self.x - self.size*2//2))
            pygame.draw.circle(s, (255,250,205, 128), [int(self.x), int(self.y)], int(self.size)*2)
            
        else:
            pygame.draw.circle(self.screen, self.color, [int(self.x), int(self.y)], int(self.size))


class Agent(Body):
    def __init__(self, screen, startx, starty, mass):
        self.screen = screen
        self.x = startx
        self.y = starty
        self.m = mass
        self.vx = 0.06
        self.vy = -0.025
        self.size = 6
        self.color = (255, 255, 255)
        self.speed = 0.001

    def render(self):
        square = pygame.Rect(self.x, self.y, self.size, self.size)
        square.move_ip(-self.size//2, -self.size//2)
        pygame.draw.rect(self.screen, self.color, square)

    def move(self, action):
        jetSize = 10 # pixels
        red = (255, 0, 0)
        if action = None:
            keys=pygame.key.get_pressed()
            if keys[K_LEFT]:
                self.vx -= self.speed
                pygame.draw.line(self.screen, red, (self.x, self.y), (self.x+jetSize, self.y))
            if keys[K_RIGHT]:
                self.vx += self.speed
                pygame.draw.line(self.screen, red, (self.x, self.y), (self.x-jetSize, self.y))
            if keys[K_DOWN]:
                self.vy += self.speed
                pygame.draw.line(self.screen, red, (self.x, self.y), (self.x, self.y-jetSize))
            if keys[K_UP]:
                self.vy -= self.speed
                pygame.draw.line(self.screen, red, (self.x, self.y), (self.x, self.y+jetSize))
        else:
            # left
            if action[0]:
                self.vx -= self.speed
                pygame.draw.line(self.screen, red, (self.x, self.y), (self.x+jetSize, self.y))
            # right
            if action[1]:
                self.vx += self.speed
                pygame.draw.line(self.screen, red, (self.x, self.y), (self.x-jetSize, self.y))
            # down
            if action[2]:
                self.vy += self.speed
                pygame.draw.line(self.screen, red, (self.x, self.y), (self.x, self.y-jetSize))
            # up
            if action[3]:
                self.vy -= self.speed
                pygame.draw.line(self.screen, red, (self.x, self.y), (self.x, self.y+jetSize))


class SolarescapeEnv(object):
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height):
        pygame.init()
        pygame.display.set_caption("Press ESC to quit")
        self.seed()
        # PyGame related
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        self.background = pygame.Surface(self.screen.get_size())
        self.background.convert() 
        self.font = pygame.font.SysFont('mono', 20, bold=True)
        # List of celestial bodies at play (including agent)
        self.bodies = []
        self.agent = None
        # Nop, fire left, right, top, or bottom thruster.
        self.action_spacce = spaces.Discrete(5)

        self.reset()


    def step(self, action):
        self.agent.move(action)
        pass

    def reset(self):
        self.bodies = []
        pass

    def render(self):
        for bod in self.bodies:
            bod.render()
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def interact(self, A, B):
        # calculate force, distance between A and B
        dx = B.x - A.x
        dy = B.y - A.y
        force = calculations.force(dx, dy, A.m, A.size, B.m, B.size)
        dist = calculations.dist(dx, dy, A.size, B.size)
        # Use those values to update acceleration, and distance btwn

        accelerationA = force / (A.m * 1000000000000) # 1000000000000
        accelerationB = force / (B.m * 1000000000000)
        if type(A) is Agent:
            accelerationA *= 3
        if type(B) is Agent:
            accelerationB *= 3
        compAx = dx / dist
        compAy = dy / dist
        compBx = -compAx
        compBy = -compAy
        A.vx += accelerationA * compAx
        A.vy += accelerationA * compAy
        B.vx += accelerationB * compBx
        B.vy += accelerationB * compBy

        A.x += A.vx
        A.y += A.vy
        B.x += B.vx
        B.y += B.vy        
        
    def move(self, agent):
        if agent == None:
            # call interact on all bodies
            numBodies = range(len(self.bodies))
            for i in numBodies:
                for j in numBodies[i+1:]:
                        self.interact(self.bodies[i], self.bodies[j])
        else:
            agent.move()

    def draw(self):
        # Call each body's render function here
        for body in self.bodies:
            body.render()

    def randrange_float(self, start, stop, step):
        return random.randint(0, int((stop - start) / step)) * step + start
        

    def placeAtRandom(self):
        for _ in range(3):
            self.bodies.append(Body(self.screen, random.randint(0,self.width), random.randint(0, self.height), 0, 0, self.randrange_float(1,10000,100), None))

    def createSystem(self):
        self.bodies.append(Body(self.screen, self.width/2, self.height/2, 0, 0, 1989, 20))
        self.bodies.append(Body(self.screen, self.width/2, self.height/2+200, 0.09, -0.001, 0.05972, 6))
        self.bodies.append(Body(self.screen, self.width/2, self.height/2+400, 0.09, -0.001, 0.05972, 6))

    def createThreeBody(self):
        self.bodies.append(Body(self.screen, self.width/2, self.height/2-173, 0.05, 0.2, 30000, None))
        self.bodies.append(Body(self.screen, self.width/2-100, self.height/2, 0.2, -0.05, 30000, None))
        self.bodies.append(Body(self.screen, self.width/2+100, self.height/2, -0.05, -0.2, 30000, None))
        

    def run(self):
        #targetPlanet = Body(self.screen,450, 450, 0, 0, 5000, None)
        #targetPlanet.target = True
        #self.bodies.append(targetPlanet)

        #self.bodies.append(Body(self.screen, 450, 600, 0.82, 0, 1, 3))
        #self.bodies.append(Body(self.screen, 450, 800, 0.82, 0, 1, 3))
        #self.bodies.append(Body(self.screen, 450, 500, 0.45, 0, 0.00000000001, 3))

        # self.bodies.append(Body(self.screen, 600, 420, 0, 0.05, 1001, 3))
        # self.bodies.append(Body(self.screen, 900, 900, 0.02, 0, 1001, 3))
        # self.bodies.append(Body(self.screen, 450, 650, .35, 0, 1, 3))

        #self.placeAtRandom()

        self.createThreeBody()

        agent = Agent(self.screen, self.width/2, self.height/2+100, 1)
        self.bodies.append(agent)
        #self.createSystem()
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
            self.move(None)
            self.move(agent)
            self.draw()
        pygame.quit()


if __name__ == '__main__':

    # call with width of window
    App(1400, 900).run()
