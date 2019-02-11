import pygame
from pygame.locals import *
 
class App(object):

    def __init__(self, width=640, height=400, fps=30):
        pygame.init()
        pygame.display.set_caption("Press ESC to quit")
        self.width = width
        self.height = height
        self.fps = fps
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.clock = pygame.time.Clock()
        self.playtime = 0.0
        self.font = pygame.font.SysFont('mono', 20, bold=True)


    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            milliseconds = self.clock.tick(self.fps)
            self.playtime += milliseconds / 1000.0

            self.drawCircle()

            pygame.display.flip()
            self.screen.blit(self.background, (0, 0))
        pygame.quit()


if __name__ == '__main__':

    # call with width of window and fps
    App(640, 400).run()