import pygame
import time

from view.utils import load_gif_frames
from core.qlearning import Launcher
from view.animatedSprite import AnimatedSprite

TILE_SIZE = 64

class Engine:
    def __init__(self, width, height):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width * TILE_SIZE, height * TILE_SIZE))
        pygame.display.set_caption("Tile Terrain Engine")

        self.sprites = {
            "corner_tl": pygame.image.load("sprites/terrains/sol_coin_gauche_haut.png"),
            "corner_tr": pygame.image.load("sprites/terrains/sol_coin_droit_haut.png"),
            "corner_bl": pygame.image.load("sprites/terrains/sol_coin_gauche_bas.png"),
            "corner_br": pygame.image.load("sprites/terrains/sol_coin_droit_bas.png"),
            "edge_top": pygame.image.load("sprites/terrains/sol_coin_haut_millieu.png"),
            "edge_bottom": pygame.image.load("sprites/terrains/sol_coin_bas_millieu.png"),
            "edge_left": pygame.image.load("sprites/terrains/sol_coin_gauche_millieu.png"),
            "edge_right": pygame.image.load("sprites/terrains/sol_coin_droit_millieu.png"),
            "center": pygame.image.load("sprites/terrains/sol.png"),
        }

        self.player = AnimatedSprite("sprites/perso/robotGame64.gif", (0, 0))

        self.enemies = [
            AnimatedSprite("sprites/perso/slime_bleu.gif",   (0, 1)),
            AnimatedSprite("sprites/perso/slime_rouge.gif",  (2, 1)),
            AnimatedSprite("sprites/perso/slime_violet.gif", (1, 3)),
            AnimatedSprite("sprites/perso/slime_violet.gif", (3, 2)),
        ]

        self.flag = AnimatedSprite("sprites/flag/flag_purple.gif", (3,3))

        self.launcher = Launcher(width , height)

        self.move_index = 0
        self.move_timer = 0
        self.play_moves = False
    
    def startQLearnin(self):
        self.launcher.test_qlearning()
        self.move_index = 0
        self.move_timer = 0
        self.play_moves = True   
    
    def draw_map(self):
        for y in range(self.height):
            for x in range(self.width):

                # Coins
                if x == 0 and y == 0:
                    sprite = self.sprites["corner_tl"]
                elif x == self.width - 1 and y == 0:
                    sprite = self.sprites["corner_tr"]
                elif x == 0 and y == self.height - 1:
                    sprite = self.sprites["corner_bl"]
                elif x == self.width - 1 and y == self.height - 1:
                    sprite = self.sprites["corner_br"]

                # Bords horizontaux
                elif y == 0:
                    sprite = self.sprites["edge_top"]
                elif y == self.height - 1:
                    sprite = self.sprites["edge_bottom"]

                # Bords verticaux
                elif x == 0:
                    sprite = self.sprites["edge_left"]
                elif x == self.width - 1:
                    sprite = self.sprites["edge_right"]

                # Centre
                else:
                    sprite = self.sprites["center"]

                self.screen.blit(sprite, (x * TILE_SIZE, y * TILE_SIZE))

    def run(self):
        running = True
        clock = pygame.time.Clock()
        move_delay = 5
    
        while running:
            dt = clock.tick(240)  # temps depuis derniere frame
           
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.startQLearnin()

            # === LECTURE DU CHEMIN Q-LEARNING ===
            if self.play_moves and self.move_index < len(self.launcher.all_moves):
                self.move_timer += dt
                if self.move_timer >= move_delay:
                    old_pos, action, new_pos = self.launcher.all_moves[self.move_index]
                    self.player.pos = new_pos 
                    self.move_index += 1
                    self.move_timer = 0

            # dessine le terrain
            self.draw_map()

            # dessine le player
            self.player.update(dt)
            self.player.draw(self.screen)

            # dessine les enemis
            for enemy in self.enemies:
                enemy.update(dt)
                enemy.draw(self.screen)

            # dessine le drapeau
            self.flag.update(dt)
            self.flag.draw(self.screen)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


    def play_moves(self):
        for (old_pos, action, new_pos) in self.launcher.all_moves:
            # Met à jour la position du joueur dans la grille
            self.player.grid_pos = new_pos  
            time.sleep(0.5)  # Une action chaque 0.5 sec


