TILE_SIZE = 64
from view.utils import load_gif_frames

class AnimatedSprite:
    def __init__(self, gif_path, pos):
        self.frames = load_gif_frames(gif_path)
        self.index = 0
        self.pos = pos  # (x, y) sur la grille
        self.time = 0   # timer interne

    def update(self, dt):
        self.time += dt
        if self.time >= 60:  # change de frame toutes les 60 ms
            self.index = (self.index + 1) % len(self.frames)
            self.time = 0

    def draw(self, screen):
        frame = self.frames[self.index]
        screen.blit(frame, (self.pos[0] * TILE_SIZE, self.pos[1] * TILE_SIZE))


