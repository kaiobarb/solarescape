import time
import torch
import cv2
from solarescape_env import *
from ple import PLE


game = SolarescapeEnv(width=856, height=856, dt=1)
game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
p = PLE(game, fps=30, frame_skip = 3, num_steps = 1,
        force_fps = False, display_screen=False)

#def getState(self):
#    state = {
#        "agent_y": self.agent.pos.y,
#        "agent_x": self.agent.pos.x,
#        "agent_velocity_x": self.agent.vel.x,
#        "agent_velocity_y": self.agent.vel.y
#    }
#    return state

def extract_image(image_data, size, tresh=True):
    #resize image and change to grayscale
    snapshot = cv2.cvtColor(cv2.resize(image_data, size), cv2.COLOR_BGR2GRAY)

    #threshold function applies fixed-level thresholding (?) to a single-channel array
    #threshold(array, threshold, maximum value, thresholding type
    if tresh:
        _, snapshot = cv2.threshold(snapshot, 100, 255, cv2.THRESH_BINARY)
    return snapshot

class LearningAgent():
    def __init__(self, actions):
        self.actions = actions

    def pickAction(self):
        return self.actions[np.random.randint(0, len(self.actions))]

if __name__ == '__main__':
    print(game.getActions())
    reward = 0
    la = LearningAgent(list(game.getActions()))
    #where is the documentation for extract_image? I imagine it comes from utils
    snapshot = extract_image(p.getScreenRGB(), (80,80), tresh=tresh)
    #what is this
    stack_snaps = np.stack((snapshot, snapshot, snapshot, snapshot), axis=0)

    while p.game_over() == False:
        snapshot = extract_image(p.getScreenRGB(), (80, 80), tresh=tresh)
        snapshot = np.reshape(snapshot, (1, 80, 80))
        st = np.append(stack_snaps[1:4, :, :], snapshot, axis=0) #what does st stand for?
        #if train goes here

    for i in range(100):
           # ob = game.init()
            while True:
                p.act(la.pickAction())
                # if (int(time.time()) % 5 == 0):
                #     action = na.pickAction()
                #     #action = na.actions[2]
                #     #print(action)
                #     p.act(action)
                # else:
                #     p.act(None)
    