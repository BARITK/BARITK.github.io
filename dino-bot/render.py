"""Optional pygame window for watching the simulator (training or a saved brain)."""

from __future__ import annotations

import pygame

from runner_env import (GROUND_Y, HEIGHT, TREX_DUCK_H, TREX_DUCK_W, TREX_H, TREX_W, TREX_X,
                        WIDTH, Dinos, World)

SCALE = 2
BG = (247, 247, 247)
INK = (83, 83, 83)
CACTUS = (60, 140, 80)
BIRD = (200, 90, 70)


class Viewer:
    def __init__(self, title: str = "dino-bot"):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH * SCALE, (HEIGHT + 30) * SCALE))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("monospace", 14 * SCALE // 2 + 4)
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.closed = False
        self.dino_layer = pygame.Surface((WIDTH * SCALE, HEIGHT * SCALE), pygame.SRCALPHA)

    def handle_events(self) -> bool:
        """Returns False if the window was closed. F = toggle fast-forward."""
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                self.closed = True
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_f:
                self.fps = 0 if self.fps else 60
        return not self.closed

    def draw(self, world: World, dinos: Dinos, hud: str = "", max_dinos: int = 60) -> bool:
        if not self.handle_events():
            return False
        s = SCALE
        self.screen.fill(BG)
        ground = (GROUND_Y + TREX_H) * s
        pygame.draw.line(self.screen, INK, (0, ground), (WIDTH * s, ground), 2)

        for o in world.obstacles:
            colour = BIRD if o.kind.name == "pterodactyl" else CACTUS
            for bx, by, bw, bh in o.boxes:
                pygame.draw.rect(self.screen, colour, ((o.x + bx) * s, (o.y + by) * s, bw * s, bh * s))

        # Draw the living dinos semi-transparent so you can see the whole population.
        self.dino_layer.fill((0, 0, 0, 0))
        alive = dinos.alive.nonzero()[0][:max_dinos]
        alpha = 255 if len(alive) <= 1 else 90
        for i in alive:
            if dinos.ducking[i]:
                rect = (TREX_X * s, (GROUND_Y + TREX_H - TREX_DUCK_H) * s, TREX_DUCK_W * s, TREX_DUCK_H * s)
            else:
                rect = (TREX_X * s, dinos.y[i] * s, TREX_W * s, TREX_H * s)
            pygame.draw.rect(self.dino_layer, (*INK, alpha), rect, border_radius=6)
            ex = (TREX_X + (TREX_DUCK_W - 10 if dinos.ducking[i] else TREX_W - 12)) * s
            ey = ((GROUND_Y + TREX_H - TREX_DUCK_H + 5) if dinos.ducking[i] else dinos.y[i] + 6) * s
            pygame.draw.rect(self.dino_layer, (*BG, alpha), (ex, ey, 4 * s, 4 * s))
        self.screen.blit(self.dino_layer, (0, 0))

        text = f"score {world.score:5d}   speed {world.speed:4.1f}   alive {int(dinos.alive.sum())}   {hud}"
        self.screen.blit(self.font.render(text, True, INK), (10, (HEIGHT + 6) * s))
        self.screen.blit(self.font.render("F: fast-forward   Esc: close", True, (150, 150, 150)),
                         (WIDTH * s - 330, (HEIGHT + 6) * s))
        pygame.display.flip()
        if self.fps:
            self.clock.tick(self.fps)
        return True

    def close(self) -> None:
        pygame.quit()
