import pygame

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

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.draw_map()
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


