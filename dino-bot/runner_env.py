"""
A fast, headless endless-runner simulator for training.

Its physics and obstacle rules use the same numbers as the dinosaur game built
into Chrome (chrome://dino): gravity, jump velocity, speed-up rate, obstacle
sizes, gaps between obstacles and collision boxes.  A bot trained here can
therefore play the real game (see chrome.py) without having to learn it from
scratch in real time.

The obstacles never depend on what the player does, so a whole population of
dinos can run in *the same world* at once.  All dino physics is vectorised with
numpy, which is what makes training fast (thousands of games per minute).

Coordinates follow the real game: a 600x150 canvas, y grows downwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# Constants (from Chrome's offline dino game)
# --------------------------------------------------------------------------- #
WIDTH, HEIGHT = 600, 150
TREX_X = 24                  # dino's x position while running (after the intro)
GROUND_Y = 93                # dino's top y when standing on the ground
TREX_W, TREX_H = 44, 47
TREX_DUCK_W, TREX_DUCK_H = 59, 25

GRAVITY = 0.6
INITIAL_JUMP_VELOCITY = -10.0
DROP_VELOCITY = -5.0
MAX_JUMP_Y = 30                          # jump is cut short above this y
MIN_JUMP_Y = GROUND_Y - 30               # must climb this far before a jump can be cut short
SPEED_DROP_COEFFICIENT = 3

START_SPEED = 6.0
MAX_SPEED = 13.0
ACCELERATION = 0.001
CLEAR_FRAMES = 180                       # 3 s with no obstacles at the start
GAP_COEFFICIENT = 0.6
MAX_GAP_COEFFICIENT = 1.5
MAX_OBSTACLE_LENGTH = 3
MAX_OBSTACLE_DUPLICATION = 2
SCORE_COEFFICIENT = 0.025                # distance -> displayed score

# Action ids the network chooses between.
NOOP, JUMP, DUCK = 0, 1, 2
N_ACTIONS = 3


@dataclass(frozen=True)
class ObstacleType:
    name: str
    width: int
    height: int
    ys: tuple[int, ...]
    multiple_speed: float        # groups of 2-3 allowed only above this speed
    min_gap: int
    min_speed: float
    boxes: tuple[tuple[int, int, int, int], ...]   # collision boxes (x, y, w, h)


OBSTACLE_TYPES = (
    ObstacleType("cactusSmall", 17, 35, (105,), 4, 120, 0,
                 ((0, 7, 5, 27), (4, 0, 6, 34), (10, 4, 7, 14))),
    ObstacleType("cactusLarge", 25, 50, (90,), 7, 120, 0,
                 ((0, 12, 7, 38), (8, 0, 7, 49), (13, 10, 10, 38))),
    ObstacleType("pterodactyl", 46, 40, (100, 75, 50), 999, 150, 8.5,
                 ((15, 15, 16, 5), (18, 21, 24, 6), (2, 14, 4, 3), (6, 10, 4, 7), (10, 8, 6, 9))),
)

RUN_BOXES = np.array([(22, 0, 17, 16), (1, 18, 30, 9), (10, 35, 14, 8),
                      (1, 24, 29, 5), (5, 30, 21, 4), (9, 34, 15, 4)], dtype=float)
# Ducking has a single box; pad with boxes far off-screen so both have 6 rows.
DUCK_BOXES = np.array([(1, 18, 55, 25)] + [(-9999, -9999, 1, 1)] * 5, dtype=float)


@dataclass
class Obstacle:
    kind: ObstacleType
    x: float
    y: int
    width: int
    height: int
    speed_offset: float
    gap: int
    boxes: np.ndarray                    # (k, 4) relative to (x, y)
    followed: bool = field(default=False)


# --------------------------------------------------------------------------- #
# Observation: the numbers the network sees.  Shared with chrome.py so the
# simulator and the real game feed the network exactly the same thing.
# --------------------------------------------------------------------------- #
N_OBSTACLES_SEEN = 2
OBS_SIZE = 3 + 5 * N_OBSTACLES_SEEN


def obstacle_features(trex_x: float, speed: float,
                      obstacles: list[tuple[float, float, float, float]]) -> np.ndarray:
    """
    obstacles: list of (x, y, width, height) in game coordinates.
    Returns 5 numbers for each of the next N_OBSTACLES_SEEN obstacles that are
    not yet behind the dino: distance, time until it arrives, top, bottom,
    width (all roughly 0..1).
    """
    ahead = sorted((o for o in obstacles if o[0] + o[2] > trex_x), key=lambda o: o[0])
    feats = []
    for i in range(N_OBSTACLES_SEEN):
        if i < len(ahead):
            x, y, w, h = ahead[i]
            dx = x - trex_x
            feats += [dx / WIDTH, dx / speed / 60.0, y / HEIGHT, (y + h) / HEIGHT, w / 100.0]
        else:
            feats += [1.5, 1.5, 0.0, 0.0, 0.0]          # "nothing coming"
    return np.array(feats)


def dino_features(y, vy, speed):
    """Height above ground, vertical velocity, game speed (vectorised)."""
    y = np.asarray(y, dtype=float)
    return np.stack([(GROUND_Y - y) / 100.0,
                     np.asarray(vy, dtype=float) / 10.0,
                     np.full_like(y, speed / MAX_SPEED)], axis=-1)


# --------------------------------------------------------------------------- #
# World: speed, distance and obstacles (shared by every dino)
# --------------------------------------------------------------------------- #
class World:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.speed = START_SPEED
        self.distance = 0.0
        self.frame = 0
        self.obstacles: list[Obstacle] = []
        self.history: list[int] = []

    @property
    def score(self) -> int:
        return math.ceil(self.distance * SCORE_COEFFICIENT)

    def move_obstacles(self) -> None:
        self.frame += 1
        if self.frame <= CLEAR_FRAMES:
            return
        for o in self.obstacles:
            o.x -= math.floor(self.speed + o.speed_offset + 1e-9)
        self.obstacles = [o for o in self.obstacles if o.x + o.width > 0]
        if not self.obstacles:
            self._add_obstacle()
        else:
            last = self.obstacles[-1]
            if not last.followed and last.x + last.width + last.gap < WIDTH:
                self._add_obstacle()
                last.followed = True

    def advance(self) -> None:
        self.distance += self.speed
        if self.speed < MAX_SPEED:
            self.speed += ACCELERATION

    def _add_obstacle(self) -> None:
        rng = self.rng
        while True:
            t = int(rng.integers(0, len(OBSTACLE_TYPES)))
            kind = OBSTACLE_TYPES[t]
            duplicate = self.history.count(t) >= MAX_OBSTACLE_DUPLICATION
            if not duplicate and self.speed >= kind.min_speed:
                break
        size = int(rng.integers(1, MAX_OBSTACLE_LENGTH + 1))
        if size > 1 and kind.multiple_speed > self.speed:
            size = 1
        width = kind.width * size
        boxes = np.array(kind.boxes, dtype=float)
        if size > 1:
            boxes[1, 2] = width - boxes[0, 2] - boxes[2, 2]
            boxes[2, 0] = width - boxes[2, 2]
        min_gap = round(width * self.speed + kind.min_gap * GAP_COEFFICIENT)
        gap = int(rng.integers(min_gap, round(min_gap * MAX_GAP_COEFFICIENT) + 1))
        offset = 0.0
        if kind.name == "pterodactyl":
            offset = 0.8 if rng.random() > 0.5 else -0.8
        y = kind.ys[int(rng.integers(0, len(kind.ys)))]
        self.obstacles.append(Obstacle(kind, float(WIDTH), y, width, kind.height, offset, gap, boxes))
        self.history = ([t] + self.history)[:MAX_OBSTACLE_DUPLICATION]

    def obstacle_tuples(self) -> list[tuple[float, float, float, float]]:
        return [(o.x, o.y, o.width, o.height) for o in self.obstacles]


# --------------------------------------------------------------------------- #
# Dinos: a vectorised batch of players
# --------------------------------------------------------------------------- #
class Dinos:
    def __init__(self, n: int):
        self.n = n
        self.y = np.full(n, float(GROUND_Y))
        self.vy = np.zeros(n)
        self.jumping = np.zeros(n, bool)
        self.ducking = np.zeros(n, bool)
        self.speed_drop = np.zeros(n, bool)
        self.reached_min = np.zeros(n, bool)
        self.alive = np.ones(n, bool)
        self.distance = np.zeros(n)          # distance travelled when they died

    def observe(self, world: World) -> np.ndarray:
        obst = obstacle_features(TREX_X, world.speed, world.obstacle_tuples())
        dino = dino_features(self.y, self.vy, world.speed)
        return np.concatenate([dino, np.broadcast_to(obst, (self.n, obst.size))], axis=1)

    def act(self, actions: np.ndarray, speed: float) -> None:
        """Apply the held action for this frame (same rules as chrome.py's driver)."""
        live = self.alive
        duck = (actions == DUCK) & live
        jump = (actions == JUMP) & live

        # Down key: duck on the ground, fast-fall in the air.
        self.ducking = duck & ~self.jumping
        start_drop = duck & self.jumping & ~self.speed_drop
        self.vy[start_drop] = 1.0
        self.speed_drop = (self.speed_drop | start_drop) & duck

        # Jump key: start a jump on the ground, releasing it cuts the jump short.
        start = jump & ~self.jumping & ~self.ducking
        self.vy[start] = INITIAL_JUMP_VELOCITY - speed / 10.0
        self.jumping |= start
        self.reached_min[start] = False
        self.speed_drop[start] = False

        released = ~jump & self.jumping & self.reached_min & (self.vy < DROP_VELOCITY)
        self.vy[released] = DROP_VELOCITY

    def update_jump(self) -> None:
        j = self.jumping & self.alive
        if not j.any():
            return
        mult = np.where(self.speed_drop, SPEED_DROP_COEFFICIENT, 1.0)
        self.y[j] += np.round(self.vy[j] * mult[j])
        self.vy[j] += GRAVITY
        self.reached_min |= j & ((self.y < MIN_JUMP_Y) | self.speed_drop)
        cut = j & ((self.y < MAX_JUMP_Y) | self.speed_drop) & self.reached_min & (self.vy < DROP_VELOCITY)
        self.vy[cut] = DROP_VELOCITY
        landed = j & (self.y > GROUND_Y)
        self.y[landed] = GROUND_Y
        self.vy[landed] = 0.0
        self.jumping[landed] = False
        self.speed_drop[landed] = False

    def collide(self, world: World) -> np.ndarray:
        """Boolean mask of dinos that hit an obstacle this frame."""
        near = [o for o in world.obstacles if o.x < TREX_X + TREX_DUCK_W and o.x + o.width > TREX_X]
        if not near:
            return np.zeros(self.n, bool)
        ob = np.concatenate([o.boxes + np.array([o.x, o.y, 0, 0]) for o in near])     # (m, 4)
        db = np.where(self.ducking[:, None, None], DUCK_BOXES, RUN_BOXES).copy()      # (n, 6, 4)
        db[..., 0] += TREX_X
        db[..., 1] += self.y[:, None]
        dx, dy, dw, dh = (db[..., i, None] for i in range(4))                        # (n, 6, 1)
        ox, oy, ow, oh = ob.T                                                         # (m,)
        hit = (dx < ox + ow) & (dx + dw > ox) & (dy < oy + oh) & (dy + dh > oy)
        return hit.any(axis=(1, 2)) & self.alive


# --------------------------------------------------------------------------- #
# Running a population through one world
# --------------------------------------------------------------------------- #
def play(policy, n: int, seed: int | None, max_score: float, on_frame=None) -> np.ndarray:
    """
    Run `n` dinos in one world until all are dead or the score reaches `max_score`.

    policy(obs, alive_idx) -> actions for the alive dinos.
    on_frame(world, dinos) is called every frame (for rendering); return False to stop.
    Returns each dino's score (same units as the score shown in Chrome).
    """
    world, dinos = World(seed), Dinos(n)
    actions = np.zeros(n, dtype=int)
    while dinos.alive.any() and world.distance * SCORE_COEFFICIENT < max_score:
        idx = np.flatnonzero(dinos.alive)
        obs = dinos.observe(world)
        actions[:] = NOOP
        actions[idx] = policy(obs[idx], idx)
        dinos.act(actions, world.speed)
        dinos.update_jump()
        world.move_obstacles()
        dead = dinos.collide(world)
        dinos.alive &= ~dead
        dinos.distance[dead] = world.distance
        world.advance()
        if on_frame is not None and on_frame(world, dinos) is False:
            break
    dinos.distance[dinos.alive] = world.distance
    return np.ceil(dinos.distance * SCORE_COEFFICIENT)
