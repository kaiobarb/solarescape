import pygame
import ple
import calculations
import sys , time , random
import numpy as np
from ple.games.base.pygamewrapper import PyGameWrapper
import math
from pygame.locals import *
from Vec2d import vec2d
from ple.games.utils import percent_round_int
from pygame.locals import *
from decimal import *
import matplotlib.pyplot as plt



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

        self.image = image.convert()
        self.rect = self.image.get_rect()
        self.rect.center = initial_position

    def draw(self, screen):
        pygame.draw.circle(
            screen,
            self.color,
            [int(self.position.x), int(self.position.y)],
            int(self.size)
        )
        screen.blit(self.image, self.rect.center)

    def interact(self, other, dt):
        dx = other.position.x - self.position.x
        dy = other.position.y - self.position.y
        force = calculations.force(dx, dy, self.mass, self.size, other.mass, other.size)
        dist = calculations.dist(dx, dy, self.size, other.size)
        acceleration = force / (self.mass*10) # 1000000000000
        compx = dx / dist
        compy = dy / dist
        self.velocity.x += acceleration * compx
        self.velocity.y += acceleration * compy
        self.position.x += self.velocity.x * dt
        self.position.y += self.velocity.y * dt
        self.rect.center = (self.position.x, self.position.y)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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
        self.velocity.x += dx * dt
        self.velocity.y += dy * dt

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

    def interact(self, other, dt):
        dx = other.position.x - self.position.x
        dy = other.position.y - self.position.y
        force = calculations.force(dx, dy, self.mass, self.size, other.mass, other.size)
        dist = calculations.dist(dx, dy, self.size, other.size)
        acceleration = force / (self.mass) # 1000000000000
        compx = dx / dist
        compy = dy / dist
        self.velocity.x += acceleration * compx
        self.velocity.y += acceleration * compy
        self.position.x += self.velocity.x * dt
        self.position.y += self.velocity.y * dt
        self.rect.center = (self.position.x, self.position.y)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class SolarescapeEnv(PyGameWrapper):
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height, dt):

        #  Action dictionary corresponding to WASD keys + a no-op
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
        self.dt = dt
        self.ticks = 0
        self.bodies = []
        self.escapes = 0
        

        """ AGENT AND SUN PROPERTIES """
        self.AGENT_COLOR = (60, 60, 140)
        self.AGENT_SPEED = 0.03
        self.AGENT_SIZE = 10
        self.AGENT_INIT_POS = (width/2, height/2+200)
        self.AGENT_MASS = int(1)

        self.SUN_COLOR = (255, 60, 60)
        self.SUN_SPEED = 0
        self.SUN_RADIUS = 20
        self.SUN_INIT_POS = (width/2, height/2)
        self.SUN_MASS = int(332953000000)

        self.dist_to_sun = 200


    # This function is called by the PLE reference to handle actions.
    # It's as if a human is pressing the keys, but not.
    def _handle_player_events(self):
        self.dx = 0.0
        self.dy = 0.0
        jetSize = 20
        centerPosition = (int(self.agent.position.x+self.agent.size/2) , int(self.agent.position.y+self.agent.size/2) )
        red = (255,0,0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == self.actions["left"]:
                    #self.score -= self.AGENT_SPEED/2
                    self.dx -= self.AGENT_SPEED
                    pygame.draw.line(self.screen, red, centerPosition, (centerPosition[0]+jetSize, centerPosition[1]))

                if key == self.actions["right"]:
                    #self.score -= self.AGENT_SPEED/2
                    self.dx += self.AGENT_SPEED
                    pygame.draw.line(self.screen, red, centerPosition, (centerPosition[0]-jetSize, centerPosition[1]))

                if key == self.actions["up"]:
                    #self.score -= self.AGENT_SPEED/2
                    self.dy -= self.AGENT_SPEED
                    pygame.draw.line(self.screen, red, centerPosition, (centerPosition[0], centerPosition[1]+jetSize))

                if key == self.actions["down"]:
                    #self.score -= self.AGENT_SPEED/2
                    self.dy += self.AGENT_SPEED
                    pygame.draw.line(self.screen, red, centerPosition, (centerPosition[0], centerPosition[1]-jetSize))
                else: 
                    pass

    # Defines the initial state of the game. This function is also used to reset the simulation.
    def init(self):
        # Agent(initial_position, color, size, speed, mass)
        self.agent = Agent(
            self.AGENT_INIT_POS,
            self.AGENT_COLOR,
            self.AGENT_SIZE,
            self.AGENT_SPEED,
            self.AGENT_MASS * self.dt
        )
        # Body(initial_position, color, radius, speed, mass)
        self.sun = Body(
            self.SUN_INIT_POS,
            self.SUN_COLOR,
            self.SUN_RADIUS,
            self.SUN_SPEED,
            self.SUN_MASS * self.dt
        )

        sun2 = Body(
            (self.width/2 -200, self.height/2+32),
            self.SUN_COLOR,
            self.SUN_RADIUS,
            self.SUN_SPEED,
            self.SUN_MASS * self.dt
        )

        self.sprite_bodies = pygame.sprite.Group()
        self.sprite_bodies.add(self.agent)
        self.sprite_bodies.add(self.sun)
        self.bodies.append(self.agent)
        self.bodies.append(self.sun)
        #self.bodies.append(sun2)
        # self.bodies.append(self.bun)
        self.score = 0
        self.ticks = 0

    ## Called every frame. 
    def step(self, action):
        # Update visual elements
        pygame.display.update()
        
        # 'ticks' is like a unity of time. Not really used, but it's here just in case.
        self.ticks += 1

        # Set background color
        self.screen.fill((0,0,0))

        # Gravity!
        for bodyA in self.bodies:
            for bodyB in self.bodies:
                if(bodyA != bodyB):
                    bodyA.interact(bodyB, self.dt)

        # Currently, rewards['tick'] = 0. This is for the case where we want reward to be 
        # passively updated over time.
        self.score = self.rewards["tick"]

        # Get distance from sun, and use that distance to reward the agent for being far from it.
        dx = self.agent.position.x - self.sun.position.x
        dy = self.agent.position.y - self.sun.position.y
        self.dist_to_sun = calculations.dist(dx, dy, self.agent.size/2, self.sun.size/2)
        if (abs(self.dist_to_sun - (self.agent.size/2 + self.sun.size/2)) < 1 ):
            self.score -= 1000
        #reward = self.dist_to_sun

        #print(self.agent.velocity.length*10, " , ", self.dist_to_sun)

        # Agent velocity is a vector, so we get the magnitude (length) of it and
        # add it to the score to reward going fast.
        reward =self.agent.velocity.length

        # Score is the actual reward that is observed by PLE, so we update that.
        self.score += reward
        #print(self.score)

        # Take action, then update the agent's position based on that action
        self._handle_player_events()
        self.agent.update(self.dx, self.dy, self.dt)

        # Draw all the agent and celestial bodies onto the screen.
        self.agent.draw(self.screen)
        for body in self.bodies:
             body.draw(self.screen)

        # Reset the simulation if the agent leaves the screen. Once we start implementing DQN we will probably want to move this step over to the other file.
        if ( self.agent.position.x > self.width or self.agent.position.x < 0 or self.agent.position.y > self.width or self.agent.position.y < 0):
            self.score += 1000
            self.escapes += 1
            self.reset()
            self.init()

        if(self.ticks > 1000):
            if self.escapes == 0:
                self.score -= 100
            self.reset()
            self.init()


        print(self.score)

    def getGameState(self):
        state = {
            "velocity_x": self.agent.velocity.x,
            "velocity_y": self.agent.velocity.y,
            "position_x": self.agent.position.x,
            "position_y": self.agent.position.y,
            "distance_to_sun": self.dist_to_sun
        }
        return state

    ## The following functions must be overriden.
    def game_over(self):
        return False

    def reset(self):
        self.bodies = []
        pass

    def getScore(self):
        return self.score


if __name__ == '__main__':

    # This part is executed if this file is executed directly. Otherwise not used.
    # Eg.: `python3 solarescape_env.py` instead of `python3 randomAgent.py`

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
