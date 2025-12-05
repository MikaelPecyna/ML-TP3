import pygame
import core.qlearning as LauncherQL
import core.deepLearning as LauncherDL
from view.animatedSprite import AnimatedSprite

TILE_SIZE = 64

class Engine:
    def __init__(self, width, height):
        pygame.init()
        self.width = width
        self.height = height
        
        self.top_margin = 40  # espace pour le texte
        self.right_margin = 200  # espace pour le texte
        self.screen = pygame.display.set_mode((width * TILE_SIZE + self.right_margin, height * TILE_SIZE + self.top_margin))

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

        self.flag = AnimatedSprite("sprites/flag/flag_purple.gif", (width - 1, height - 1))

        self.launcherQL = LauncherQL.LauncherQL(width , height)
        self.launcherDL = LauncherDL.LauncherDL(width , height)
        self.launcher = self.launcherDL #default

        self.move_index = 0
        self.move_timer = 0
        self.play_moves = False
        self.move_delay = 5  

        self.play_test = False
        self.test_path = []
        self.test_total_reward = 0
        self.test_steps = 0
     
        self.font = pygame.font.Font(None, 20)
        self.score = 0
        self.current_epoch = 0
     
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

                self.screen.blit(sprite, (x * TILE_SIZE, y * TILE_SIZE + self.top_margin))


    def draw_text(self):
        messages = []
        if self.play_moves:
            # reset
            pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.width * TILE_SIZE + self.right_margin, self.top_margin))
        
            score_text = self.font.render(f"Score: {self.score} | ", True, (255, 255, 255))
            speed_text = self.font.render(f"Delay: {self.move_delay}ms |", True, (255, 255, 255))
            epoch_text = self.font.render(f"Epoch: {self.current_epoch}/{self.launcher.max_epoch}", True, (255, 255, 255))

            self.screen.blit(score_text, (10, 5))
            self.screen.blit(speed_text, (100, 5))
            self.screen.blit(epoch_text, (200, 5)) 
        else:
            if not self.launcher.all_moves: 
                messages.append("Press SPACE to start Q-Learning")
            else:
                messages.append("Q-Learning Finished - Press SPACE to Replay")
                messages.append("Press DOWN ARROW to test")
            
            for i, msg in enumerate(messages):
                text = self.font.render(msg, True, (255, 255, 255))
                text_rect = text.get_rect(
                    center=(self.width * TILE_SIZE // 2,
                            self.height * TILE_SIZE // 2 + i * 25) 
                )
                self.screen.blit(text, text_rect)

    def run(self):
        running = True
        clock = pygame.time.Clock()
    
        while running:
            dt = clock.tick(240)  # temps depuis derniere frame
           
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.launcher.all_moves.clear()
                    self.launcher.epoch_scores.clear()
                    self.launcher.epoch_steps.clear()
                    self.move_index = 0
                    self.move_timer = 0
                    self.play_moves = True
                    self.launcher.launch_training()
                                    
                if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                    self.move_index = max(0, len(self.launcher.all_moves) - 1)


                if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                    self.launcher.launch_test()
                    self.test_path = self.launcher.test_path
                    self.test_total_reward = self.launcher.test_total_reward
                    self.test_steps = self.launcher.test_steps

                    self.move_index = 0
                    self.move_timer = 0
                    self.play_test = True
                    self.play_moves = False
                    pass

                # Control speed with arrow keys
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        self.move_delay = max(1, self.move_delay - 10)  # Augmente vitesse (diminue délai)
                    elif event.key == pygame.K_LEFT:
                        self.move_delay += 10  # Diminue vitesse (augmente délai)

            # === LECTURE DU CHEMIN Q-LEARNING ===
            if self.play_moves and self.move_index < len(self.launcher.all_moves):
                self.move_timer += dt
                if self.move_timer >= self.move_delay:
                    old_pos, action, new_pos = self.launcher.all_moves[self.move_index]
                    self.player.pos = new_pos 
                    self.move_index += 1
                    self.move_timer = 0

                    # #mise à jour du score et de l'époque
                    move_count = 0
                    for i, ep_scores in enumerate(self.launcher.epoch_scores):
                        if self.move_index <= move_count + len(ep_scores):
                            # move_index appartient à cet épisode
                            self.current_epoch = i + 1

                            # Score cumulé jusqu'à ce move
                            self.score = sum(ep_scores[:self.move_index - move_count])
                            break
                        move_count += len(ep_scores)

            if self.play_test and self.move_index < len(self.test_path):
                self.move_timer += dt
                if self.move_timer >= self.move_delay:
                    new_pos = self.test_path[self.move_index]
                    self.player.pos = new_pos
                    self.move_index += 1
                    self.move_timer = 0

            # dessine le terrain
            self.draw_map()

            # dessine le player
            self.player.update(dt)
            self.player.draw(self.screen, self.top_margin)

            # dessine les enemis
            for enemy in self.enemies:
                enemy.update(dt)
                enemy.draw(self.screen, self.top_margin)

            # dessine le drapeau
            self.flag.update(dt)
            self.flag.draw(self.screen, self.top_margin)

            # dessine le texte (epoch et score)
            self.draw_text()

            if self.play_moves and self.move_index >= len(self.launcher.all_moves):
                self.play_moves = False  
            
            if self.play_test and self.move_index >= len(self.test_path):
                self.play_test = False  

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()




