from PIL import Image
import pygame 

def load_gif_frames(path):
    pil_img = Image.open(path)
    frames = []
    try:
        while True:
            frame = pil_img.convert("RGBA")
            frame = pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode)
            frames.append(frame)
            pil_img.seek(pil_img.tell() + 1)
    except EOFError:
        pass
    return frames