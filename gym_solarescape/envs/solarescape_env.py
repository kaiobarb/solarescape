import pygame
import calculations
import sys , time , random
import numpy as np
import gym

from gym import spaces
from gym.utils import seeding
import math
from .utils.vec2d import vec2d
from .util import percent_round_int
from pygame.locals import *
from decimal import *


from base import PyGameWrapper
from pygame.constants import K_w, K_a, K_s, K_d

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

class Body(pygame.sprite.Sprite):
    def __init__(self, initial_position, color, radius, speed, mass):
        self.position = vec2d(initial_position)
        self.color = color
        self.size = size
        self.mass = mass
        self.velocity = vec2d((0,0))

        image = pygame.Surface([radius * 2, radius * 2])
        image.set_colorkey((0, 0, 0))

        pygame.draw.circle(
            image,
            color,
            initial_position,
            radius,
            0
        )

        self.image = image.convert()
        self.rect = self.image.get_rect()

        def update(self, dx, dy, dt):
            self.vel.x += dx
            self.vel.y += dy

            new_x = self.pos.x + self.vel.x * dt
            new_y = self.pos.y + self.vel.y * dt

        def draw(self, screen):
            screen.blit(self.image, self.rect.center)


class Agent(Body):
    def __init__(self, initial_position, color, size, speed, mass):
        self.position = vec2d(initial_position)
        self.color = color
        self.size = size
        self.mass = mass
        self.velocity = vec2d((0,0))

        image = pygame.Surface([size, size])
        image.set_colorkey((0, 0, 0))

        square = pygame.Rect(self.position.x, self.position.y, self.size, self.size)

        pygame.draw.rect(
            image,
            color,
            square,
            0
        )

        self.image = image.convert()
        self.rect = self.image.get_rect()

        def update(self, dx, dy, dt):
            self.vel.x += dx
            self.vel.y += dy

            # new_x = self.pos.x + self.vel.x * dt
            # new_y = self.pos.y + self.vel.y * dt

        def draw(self, screen):
            screen.blit(self.image, self.rect.center)

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

class SolarescapeEnv(PyGameWrapper):
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height):

        #  Definitions for constants used in our agent methods
        actions = {
            "up": K_w,
            "left": K_a,
            "right": K_d,
            "down": K_s
        }

        PyGameWrapper.__init__(self, width, height, actions=actions)

        self.dx = 0
        self.dy = 0
        self.ticks = 0

        self.AGENT_COLOR = (60, 60, 140)
        self.AGENT_SPEED = 0.02
        self.AGENT_RADIUS = percent_round_int(width, 0.047)
        self.AGENT_INIT_POS = (width/2, height/2+200)
        self.AGENT_MASS = 10

        self.SUN_COLOR = (255, 60, 60)
        self.SUN_SPEED = 0
        self.SUN_RADIUS = percent_round_int(width, 0.15)
        self.SUN_INIT_POS = (width/2, height/2)
        self.SUN_MASS = 1000

        def _handle_player_events(self):
            self.dx = 0.0
            self.dy = 0.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    key = event.key

                    if key == self.actions["left"]:
                        self.dx -= self.AGENT_SPEED

                    if key == self.actions["right"]:
                        self.dx += self.AGENT_SPEED

                    if key == self.actions["up"]:
                        self.dy -= self.AGENT_SPEED

                    if key == self.actions["down"]:
                        self.dy += self.AGENT_SPEED

    def step(self, action):
        self.agent.move(action)
        pass

    def init(self):
        # initial_position, color, size, speed, mass
        self.agent = Agent(
            self.AGENT_INIT_POS,
            self.AGENT_COLOR,
            self.AGENT_RADIUS,
            self.AGENT_SPEED,
            self.AGENT_MASS
        )

        self.sun = Body(
            self.AGENT_INIT_POS,
            self.AGENT_COLOR,
            self.AGENT_RADIUS,
            self.AGENT_SPEED,
            self.AGENT_MASS
        )
        self.sprite_bodies = pygame.sprite.Group()
        self.sprite_bodies.add(self.good_creep)
        self.sprite_bodies.add(self.bad_creep)
        self.score = 0
        self.ticks = 0

    def reset(self):
        self.bodies = []
        pass

    def render(self):
        for bod in self.bodies:
            bod.render()
        pass        
        
    def move(self, agent):
        if agent == None:
            # call interact on all bodies
            numBodies = range(len(self.bodies))
            for i in numBodies:
                for j in numBodies[i+1:]:
                        self.interact(self.bodies[i], self.bodies[j])
        else:
            agent.move()


    def run(self):

        self.createThreeBody()

        agent = Agent(self.screen, self.width/2, self.height/2+100, 1)
        self.bodies.append(agent)
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
