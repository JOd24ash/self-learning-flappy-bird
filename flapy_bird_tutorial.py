# ------------------------------------------------------------
# Flappy Bird played by an AI using NEAT (NeuroEvolution of
# Augmenting Topologies). Many birds play at once, each one
# controlled by its own small neural network. Birds that survive
# longer get a higher "fitness", and the best networks are
# bred to make the next generation.
# ------------------------------------------------------------

import pygame   # game graphics, images, collision masks
import neat     # neat-python library for the evolution part
import time
import os       # for building file paths
import random   # for random pipe heights

pygame.font.init()  # must be called before creating any font

# ---------------- Window and image setup ----------------
WIN_WIDTH = 600
WIN_HEIGHT = 800

# Load the 3 bird frames (wings up / middle / down) and double their size.
BIRD_IMGS = [
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png"))),
]
PIPE_IMGS = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMGS = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMGS = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

STAT_FONT = pygame.font.SysFont("comicsans", 50)  # font for the score text


# ======================= BIRD =======================
class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25    # max upward tilt in degrees
    ROT_VEL = 20         # how fast the bird tilts down
    ANIMATION_TIME = 5   # frames each wing image is shown

    def __init__(self, x, y):
        self.x = x                # horizontal position (never changes)
        self.y = y                # vertical position
        self.tilt = 0             # current rotation angle
        self.tick_count = 0       # frames since last jump (used for physics)
        self.vel = 0              # vertical velocity
        self.height = self.y      # y position where the last jump started
        self.img_count = 0        # counter for wing animation
        self.img = self.IMGS[0]   # current image

    def jump(self):
        self.vel = -10.5          # negative = upward in pygame
        self.tick_count = 0       # restart the physics timer
        self.height = self.y      # remember where we jumped from

    def move(self):
        self.tick_count += 1

        # Physics: displacement = v*t + 1.5*t^2 (gravity pulls the bird down)
        d = self.vel * self.tick_count + 1.5 * self.tick_count ** 2

        if d >= 16:               # terminal velocity: don't fall faster than 16 px/frame
            d = 16
        if d < 0:                 # moving up: give a little extra boost
            d -= 2

        self.y = self.y + d

        # Tilt: nose up while rising (or just after), nose down while falling
        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        # Cycle through wing images: 0 -> 1 -> 2 -> 1 -> 0 ...
        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        # When nose-diving, don't flap the wings
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        # Rotate the image around its center (pygame rotates around top-left by default)
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(
            center=self.img.get_rect(topleft=(self.x, self.y)).center
        )
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        # Mask = pixel-perfect shape of the bird, used for exact collision detection
        return pygame.mask.from_surface(self.img)


# ======================= PIPE =======================
class Pipe:
    GAP = 200   # vertical space between top and bottom pipe
    VEL = 5     # how fast pipes move left

    def __init__(self, x):
        self.x = x
        self.height = 0       # y where the top pipe ends (gap starts)
        self.top = 0          # y position to draw the top pipe image
        self.bottom = 0       # y position to draw the bottom pipe image
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMGS, False, True)  # upside-down pipe
        self.PIPE_BOTTOM = PIPE_IMGS
        self.passed = False   # has the bird already passed this pipe?
        self.set_height()

    def set_height(self):
        # Pick a random gap position
        self.height = random.randrange(50, 450)
        # Top pipe image is tall, so shift it up so its bottom edge sits at self.height
        self.top = self.height - self.PIPE_TOP.get_height()
        # Bottom pipe starts GAP pixels below the top pipe's end
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        # Build masks for pixel-perfect collision
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        # Offset = how far the pipe is from the bird (needed by mask.overlap)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        # overlap() returns the collision point, or None if there is none
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        return False


# ======================= BASE (ground) =======================
class Base:
    VEL = 5                          # same speed as pipes so it looks consistent
    WIDTH = BASE_IMGS.get_width()
    IMG = BASE_IMGS

    def __init__(self, y):
        self.y = y
        # Two copies of the ground image placed side by side
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        # When one image goes fully off-screen, put it behind the other.
        # This makes the ground scroll forever.
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


# ======================= DRAWING =======================
def draw_window(win, birds, pipes, base, score):
    win.blit(BG_IMGS, (0, 0))           # background first (everything draws on top)

    for pipe in pipes:
        pipe.draw(win)

    # Score in the top-right corner
    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    base.draw(win)

    for bird in birds:
        bird.draw(win)

    pygame.display.update()             # show the finished frame


# ======================= FITNESS FUNCTION =======================
# NEAT calls this once per generation. It gets ALL genomes of that
# generation, and we must set g.fitness for each one.
def main(genomes, config):
    nets = []    # neural network for each bird
    ge = []      # genome for each bird (to update its fitness)
    birds = []   # the bird objects

    # The three lists are kept index-aligned: nets[i], ge[i], birds[i]
    # all belong to the same bird.
    for _, g in genomes:                                   # genomes = (id, genome) pairs
        net = neat.nn.FeedForwardNetwork.create(g, config) # build network from genome
        nets.append(net)
        birds.append(Bird(230, 350))                       # every bird starts at same spot
        g.fitness = 0                                      # start with zero fitness
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(700)]

    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    score = 0

    run = True
    while run:
        clock.tick(30)   # limit to 30 FPS

        # Allow closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # ---- Decide which pipe the birds should look at ----
        # Normally the first pipe. Once the birds pass it, look at the next one.
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            break   # all birds are dead -> end this generation

        # ---- Move each bird and ask its network whether to jump ----
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1    # small reward for every frame alive

            # Network inputs: bird's y, distance to top pipe, distance to bottom pipe
            output = nets[x].activate((
                bird.y,
                abs(bird.y - pipes[pipe_ind].height),
                abs(bird.y - pipes[pipe_ind].bottom),
            ))

            # tanh output is between -1 and 1; jump if above 0.5
            if output[0] > 0.5:
                bird.jump()

        base.move()

        # ---- Pipes: collisions, scoring, removal ----
        add_pipe = False   # becomes True when a bird passes a pipe
        rem = []           # pipes that went off-screen
        for pipe in pipes:
            # Loop backwards so removing a bird doesn't skip the next one
            for x in range(len(birds) - 1, -1, -1):
                if pipe.collide(birds[x]):
                    ge[x].fitness -= 1   # penalty for hitting a pipe
                    birds.pop(x)         # remove bird AND its net AND its genome
                    nets.pop(x)
                    ge.pop(x)
                    continue

                # Bird has gone past this pipe
                if not pipe.passed and pipe.x < birds[x].x:
                    pipe.passed = True
                    add_pipe = True

            # Pipe fully off the left side of the screen
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5           # reward birds that are still alive
            pipes.append(Pipe(700))      # spawn the next pipe

        for r in rem:
            pipes.remove(r)

        # ---- Birds that hit the ground or fly off the top die ----
        for x in range(len(birds) - 1, -1, -1):
            if birds[x].y + birds[x].img.get_height() >= 730 or birds[x].y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        draw_window(win, birds, pipes, base, score)


# ======================= NEAT SETUP =======================
def run(config_path):
    # Load NEAT settings (population size, mutation rates, inputs/outputs, etc.)
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    p = neat.Population(config)                  # create the first random generation

    p.add_reporter(neat.StdOutReporter(True))    # print progress in the console
    p.add_reporter(neat.StatisticsReporter())    # collect stats per generation

    # Run up to 50 generations, calling main() each time to score the birds
    winner = p.run(main, 50)
    


# ======================= ENTRY POINT =======================
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)   # folder this file is in
    config_path = os.path.join(local_dir, "config_feedforward_68c3f3d4a7.txt")
    run(config_path)




