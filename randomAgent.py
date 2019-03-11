import gym
from solarescape_env import *
from ple import PLE

game = SolarescapeEnv(width=556, height=556)
game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
p = PLE(game, fps=30, display_screen=False)

class NaiveAgent():
    def __init__(self, actions):
        self.actions = actions
    def pickAction(self):
        return self.actions[np.random.randint(0, len(self.actions))]

if __name__ == '__main__':
    print(game.getActions())
    na = NaiveAgent(list(game.getActions()))
    for i in range(100):
           # ob = game.init()
            while True:
                action = na.pickAction()
                p.act(action)
                #game.step(action)
                #ob, reward, done, _ = env.step(action)
    