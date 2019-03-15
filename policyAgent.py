import time
import calculations
from solarescape_env import *
from ple import PLE


game = SolarescapeEnv(width=856, height=856, dt=5)
game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
p = PLE(game, fps=30, display_screen=False)

class PolicyAgent():
    def __init__(self, actions):
        self.actions = actions
    def getState(self):
        state = {
            "agent_y": self.agent.pos.y,
            "agent_x": self.agent.pos.x,
            "agent_velocity_x": self.agent.vel.x,
            "agent_velocity_y": self.agent.vel.y
        }
        return state
    def pickAction(self):
        state = getState()
        if
            (return self.actions[np.random.randint(0, len(self.actions))]

if __name__ == '__main__':
    print(game.getActions())
    na = NaiveAgent(list(game.getActions()))
    for i in range(100):
           # ob = game.init()
            while True:
                p.act(na.pickAction())
                # if (int(time.time()) % 5 == 0):
                #     action = na.pickAction()
                #     #action = na.actions[2]
                #     #print(action)
                #     p.act(action)
                # else:
                #     p.act(None)
    