import pygame
import ple
import calculations
import sys , time , random
import numpy as np
import gym
#from base import PyGameWrapper
from ple.games.base.pygamewrapper import PyGameWrapper
from gym import spaces
from gym.utils import seeding
import math
from pygame.locals import *
from Vec2d import vec2d
from ple.games.utils import percent_round_int
from pygame.locals import *
from decimal import *



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
        super().__init__()
        self.position = vec2d(initial_position)
        self.color = color
        self.size = radius
        self.mass = mass
        self.velocity = vec2d((0,0))

        image = pygame.Surface([self.size * 2, self.size * 2])
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        # pygame.draw.circle(
        #     image,
        #     color,
        #     initial_position,
        #     radius,
        #     0
        # )

        self.image = image.convert()
        self.rect = self.image.get_rect()
        self.rect.center = initial_position

    # def update(self, dx, dy, dt):
    #     self.vel.x += dx
    # self.vel.y += dy

    def draw(self, screen):
        pygame.draw.circle(
            screen,
            self.color,
            [int(self.position.x), int(self.position.y)],
            int(self.size)
            
        )
        screen.blit(self.image, self.rect.center)

    def interact(self, other):
        dx = other.position.x - self.position.x
        dy = other.position.y - self.position.y
        force = calculations.force(dx, dy, self.mass, self.size, other.mass, other.size)
        dist = calculations.dist(dx, dy, self.size, other.size)
        acceleration = force / (self.mass * 10000000) # 1000000000000
        compx = dx / dist
        compy = dy / dist
        self.velocity.x += acceleration * compx
        self.velocity.y += acceleration * compy
        self.position.x += self.velocity.x
        self.position.y += self.velocity.y
        self.rect.center = (self.position.x, self.position.y)


class Agent(pygame.sprite.Sprite):
    def __init__(self, initial_position, color, size, speed, mass):
        super().__init__()
        self.position = vec2d(initial_position)
        self.color = color
        self.size = size
        self.mass = mass
        self.velocity = vec2d((0,0))

        image = pygame.Surface((size, size))
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
        self.rect.center = initial_position

    def update(self, dx, dy, dt):
        # self.vel.x += dx
        # self.vel.y += dy
        self.velocity.x += dx
        self.velocity.y += dy
        #self.rect.center = (self.position.x, self.position.y)

        # new_x = self.pos.x + self.vel.x * dt
        # new_y = self.pos.y + self.vel.y * dt

    def draw(self, screen):
        image = pygame.Surface((self.size, self.size))
        #image.set_colorkey((0, 0, 0))
        #image.fill((255, 255, 255))
        square = pygame.Rect(self.position.x, self.position.y, self.size, self.size)
        pygame.draw.rect(
            screen,
            self.color,
            square,
            0
        )
        screen.blit(self.image, self.rect.center)

    def interact(self, other):
        dx = other.position.x - self.position.x
        dy = other.position.y - self.position.y
        force = calculations.force(dx, dy, self.mass, self.size, other.mass, other.size)
        dist = calculations.dist(dx, dy, self.size, other.size)
        acceleration = force / (self.mass) # 1000000000000
        compx = dx / dist
        compy = dy / dist
        self.velocity.x += acceleration * compx
        self.velocity.y += acceleration * compy
        self.position.x += self.velocity.x
        self.position.y += self.velocity.y
        self.rect.center = (self.position.x, self.position.y)

class SolarescapeEnv(PyGameWrapper):
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height):

        #  Definitions for constants used in our agent methods
        actions = {
            "up": K_w,
            "left": K_a,
            "right": K_d,
            "down": K_s,
            "nop": None
        }

        PyGameWrapper.__init__(self, width, height, actions=actions)

        self.dx = 0
        self.dy = 0
        self.ticks = 0
        self.bodies = []

        self.AGENT_COLOR = (60, 60, 140)
        self.AGENT_SPEED = 0.1
        self.AGENT_RADIUS = 10
        self.AGENT_INIT_POS = (width/2, height/2+200)
        self.AGENT_MASS = int(10)

        self.SUN_COLOR = (255, 60, 60)
        self.SUN_SPEED = 0
        self.SUN_RADIUS = 20
        self.SUN_INIT_POS = (width/2+100, height/2-100)
        self.SUN_MASS = int(100000000000)

    def _handle_player_events(self):
        self.dx = 0.0
        self.dy = 0.0
        jetSize = 10
        centerPosition = (int(self.agent.position.x+self.agent.size/2) , int(self.agent.position.y+self.agent.size/2) )
        red = (255,0,0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        #      if keys[K_LEFT]:
        #     self.vx -= self.speed
        #     pygame.draw.line(self.screen, red, (self.x, self.y), (self.x+jetSize, self.y))
        # if keys[K_RIGHT]:
        #     self.vx += self.speed
        #     pygame.draw.line(self.screen, red, (self.x, self.y), (self.x-jetSize, self.y))
        # if keys[K_DOWN]:
        #     self.vy += self.speed
        #     pygame.draw.line(self.screen, red, (self.x, self.y), (self.x, self.y-jetSize))
        # if keys[K_UP]:
        #     self.vy -= self.speed
        #     pygame.draw.line(self.screen, red, (self.x, self.y), (self.x, self.y+jetSize))

            if event.type == pygame.KEYDOWN:
                key = event.key
                #print("key: ", key)

                if key == self.actions["left"]:
                    self.dx -= self.AGENT_SPEED
                    pygame.draw.line(self.screen, red, centerPosition, (centerPosition[0]+jetSize, centerPosition[1]))

                if key == self.actions["right"]:
                    self.dx += self.AGENT_SPEED
                    pygame.draw.line(self.screen, red, centerPosition, (centerPosition[0]-jetSize, centerPosition[1]))

                if key == self.actions["up"]:
                    self.dy -= self.AGENT_SPEED
                    pygame.draw.line(self.screen, red, centerPosition, (centerPosition[0], centerPosition[1]-jetSize))

                if key == self.actions["down"]:
                    self.dy += self.AGENT_SPEED
                    pygame.draw.line(self.screen, red, centerPosition, (centerPosition[0], centerPosition[1]+jetSize))
                else: 
                    pass

    def step(self, action):
        pygame.display.update()
        self.ticks += 1
        self.screen.fill((0,0,0))
        self._handle_player_events()
        # numBodies = range(len(self.bodies))
        #print(self.bodies)
        for bodyA in self.bodies:
            for bodyB in self.bodies:
                if(bodyA != bodyB):
                    #print(bodyA.position)
                    bodyA.interact(bodyB)
        self.agent.update(self.dx, self.dy, 1)
        self.score = self.rewards["tick"]
        dx = self.agent.position.x - self.sun.position.x
        dy = self.agent.position.y - self.sun.position.y
        if (dx * dx + dy + dy) < 0.1:
            dist_to_sun = 0.1
        else:
            dist_to_sun = math.sqrt(dx * dx + dy + dy)
        reward = -dist_to_sun
        self.score += reward

        self.agent.draw(self.screen)
        for body in self.bodies:
             body.draw(self.screen)

        #if self.agent.position.x > self.width || self.agent.position.x 

    def init(self):
        # initial_position, color, size, speed, mass

        self.agent = Agent(
            self.AGENT_INIT_POS,
            self.AGENT_COLOR,
            self.AGENT_RADIUS,
            self.AGENT_SPEED,
            self.AGENT_MASS
        )
        # initial_position, color, radius, speed, mass
        self.sun = Body(
            self.SUN_INIT_POS,
            self.SUN_COLOR,
            self.SUN_RADIUS,
            self.SUN_SPEED,
            self.SUN_MASS
        )

        self.sprite_bodies = pygame.sprite.Group()
        self.sprite_bodies.add(self.agent)
        self.sprite_bodies.add(self.sun)
        self.bodies.append(self.agent)
        self.bodies.append(self.sun)
        self.score = 0
        self.ticks = 0

    def game_over(self):
        return False

    def reset(self):
        self.bodies = []
        pass

    def getScore(self):
        return self.score


if __name__ == '__main__':

    # call with width of window
    # SolarescapeEnv(1400, 900)
    # import numpy as np

    pygame.init()
    game = SolarescapeEnv(width=256, height=256)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.rng = np.random.RandomState(24)
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(60)
        game.step(dt)
        pygame.display.update()
    pass
