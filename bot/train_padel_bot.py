"""Train a neural-network bot for Padel Bounce (padel.html) with evolution strategies.

The simulator below is a line-by-line port of the game's fixed-step (60 Hz) physics,
in units where the screen height H = 1 and the width W = aspect ratio (W/H).
Run:  python3 train_padel_bot.py  [generations]
Writes padel-bot-weights.json next to this file.
"""
import json, sys, time
import numpy as np

G0 = 0.0009
N_IN, N_H, N_OUT = 14, 32, 2
PX = 1 / 844  # one pixel on a typical phone, used for the game's "2px" nudge


class Env:
    """N games run in parallel (vectorised with numpy)."""

    def __init__(self, n, rng, aspect=None):
        self.n, self.rng, self.aspect = n, rng, aspect
        self.reset()

    def reset(self):
        n, r = self.n, self.rng
        z = np.zeros(n)
        self.W = np.full(n, self.aspect) if self.aspect else r.uniform(0.42, 0.75, n)
        m = np.minimum(self.W, 1.0)
        self.R, self.br = m * 0.11, m * 0.028
        side = r.choice([-1.0, 1.0], n)
        self.bx = self.W / 2 + side * self.W * 0.18
        self.by = z + 0.2
        self.bvx = -side * self.W * 0.004
        self.bvy = z.copy()
        self.rx, self.ry = self.W / 2, z + 0.72
        self.rpx, self.rpy = self.rx.copy(), self.ry.copy()
        self.rvx, self.rvy = z.copy(), z.copy()
        self.score, self.t = z.copy(), 0
        self.armed = np.zeros(n, bool)
        self.lastHitX = np.full(n, np.nan)
        self.ev = np.zeros(n, int)          # 0 none, 1 wind, 2 gold
        self.evT, self.wind, self.gust = z.copy(), z.copy(), z.copy()
        self.nextAt = z + 6
        self.done = np.zeros(n, bool)
        self.steps = np.zeros(n, int)

    def obs(self):
        W = self.W
        tier = np.floor(self.score / 50)
        return np.stack([
            self.bx / W * 2 - 1, self.by * 2 - 1, self.bvx * 40, self.bvy * 40,
            self.rx / W * 2 - 1, self.ry * 2 - 1, self.rvx * 40, self.rvy * 40,
            (self.bx - self.rx) / W * 2, (self.by - self.ry) * 2,
            self.wind / (W * 0.007), (self.ev == 2) * 1.0, tier * 0.55, W * 2 - 1,
        ], 1)

    def step(self, ax, ay):
        r, W, R, br = self.rng, self.W, self.R, self.br
        alive = ~self.done
        self.t += 1
        # pointer -> racket (same easing and clamps as the game)
        px = self.bx + ax * W * 0.5
        py = 0.5 + (ay + 1) * 0.25
        tx = np.clip(px, R * 0.6, W - R * 0.6)
        ty = np.clip(py, 0.5 + R, 1 - R * 0.6)
        self.rx = self.rx + (tx - self.rx) * 0.75
        self.ry = self.ry + (ty - self.ry) * 0.75
        self.rvx, self.rvy = self.rx - self.rpx, self.ry - self.rpy
        self.rpx, self.rpy = self.rx.copy(), self.ry.copy()

        tier = np.floor(self.score / 50)
        g = G0 * (1 + tier * 0.55)
        self.bvy = self.bvy + g
        # wind
        wa = self.wind != 0
        flip = wa & (self.gust != 0) & (self.evT < self.gust)
        self.wind = np.where(flip, -self.wind, self.wind)
        self.gust = np.where(flip, self.evT - (50 + r.random(self.n) * 60), self.gust)
        w = self.wind * (0.7 + 0.3 * np.sin(self.t * 0.12))
        self.bvx = np.where(wa, self.bvx + (w - self.bvx) * 0.05, self.bvx)

        self.bx = self.bx + self.bvx
        self.by = self.by + self.bvy
        # event timer
        act = self.ev != 0
        self.evT = np.where(act, self.evT - 1, self.evT)
        end = act & (self.evT <= 0)
        self.ev = np.where(end, 0, self.ev)
        self.wind = np.where(end, 0.0, self.wind)
        # side walls
        lo = self.bx < br
        self.bx = np.where(lo, br, self.bx); self.bvx = np.where(lo, np.abs(self.bvx) * 0.95, self.bvx)
        hi = self.bx > W - br
        self.bx = np.where(hi, W - br, self.bx); self.bvx = np.where(hi, -np.abs(self.bvx) * 0.95, self.bvx)
        # ceiling (+5 once per hit)
        top = self.by < br
        self.by = np.where(top, br, self.by)
        self.bvy = np.where(top, np.abs(self.bvy) * 0.6, self.bvy)
        roof = top & self.armed & alive
        self.score += roof * 5.0
        self.armed &= ~top
        self.bvx = self.bvx * 0.999

        # racket hit
        nx = (self.bx - self.rx) / (R * 1.05)
        ny = (self.by - self.ry) / (R * 0.9)
        hit = (nx * nx + ny * ny <= 1.15) & (self.bvy > 0) & (self.by <= self.ry + br) & alive
        if hit.any():
            kick = 0.5 + tier * 0.22
            up = np.maximum(0, -self.rvy)
            base = np.sqrt(2 * g * 0.5)
            nbvy = -(base * 0.95 + np.maximum(0, up - 0.006))
            rnd_dir = r.choice([-1.0, 1.0], self.n)
            d = np.where(nx != 0, np.sign(nx), rnd_dir)
            travel = np.where(np.isnan(self.lastHitX), 0.0, np.sign(self.rx - np.nan_to_num(self.lastHitX)))
            away = np.where(self.rx < W * 0.3, 1.0, np.where(self.rx > W * 0.7, -1.0, np.where(travel != 0, travel, d)))
            nbvx = (self.bvx * 0.25 + nx * 0.0045 + self.rvx * 0.45 + away * 0.0035 * kick
                    + (r.random(self.n) - 0.5) * 0.0025 * kick + self.wind * 1.6)
            self.bvy = np.where(hit, nbvy, self.bvy)
            self.bvx = np.where(hit, nbvx, self.bvx)
            self.armed |= hit
            self.lastHitX = np.where(hit, self.rx, self.lastHitX)
            self.by = np.where(hit, self.ry - br - 2 * PX, self.by)
            self.score += hit * np.where(self.ev == 2, 2.0, 1.0)
            start = hit & (self.score >= self.nextAt) & (self.ev == 0)
            if start.any():
                typ = r.choice([1, 2], self.n)
                self.ev = np.where(start, typ, self.ev)
                self.evT = np.where(start, 300.0, self.evT)
                sgn = r.choice([-1.0, 1.0], self.n)
                isw = start & (typ == 1)
                self.wind = np.where(isw, sgn * W * 0.007, np.where(start, 0.0, self.wind))
                self.gust = np.where(isw, 300 - (40 + r.random(self.n) * 60), np.where(start, 0.0, self.gust))
                self.nextAt = np.where(start, self.score + 5 + r.integers(0, 5, self.n), self.nextAt)
        # game over
        self.steps += alive
        self.done |= (self.by - br > 1)


def n_params():
    return N_IN * N_H + N_H + N_H * N_H + N_H + N_H * N_OUT + N_OUT


def unpack(theta):
    """theta: (P, n_params) -> list of weight arrays with leading P axis."""
    P = theta.shape[0]; i = 0; out = []
    for shape in [(N_IN, N_H), (N_H,), (N_H, N_H), (N_H,), (N_H, N_OUT), (N_OUT,)]:
        k = int(np.prod(shape)); out.append(theta[:, i:i + k].reshape(P, *shape)); i += k
    return out


def policy(ws, obs):
    """obs: (P, E, N_IN)"""
    W1, b1, W2, b2, W3, b3 = ws
    h = np.tanh(np.einsum('pei,pij->pej', obs, W1) + b1[:, None])
    h = np.tanh(np.einsum('pei,pij->pej', h, W2) + b2[:, None])
    return np.tanh(np.einsum('pei,pij->pej', h, W3) + b3[:, None])


def rollout(theta, episodes, max_steps, seed, aspect=None):
    """Return mean score per parameter vector (common random numbers across members)."""
    P = theta.shape[0]
    rng = np.random.default_rng(seed)
    env = Env(P * episodes, rng, aspect)
    if aspect is None:  # same screen sizes for every member
        env.W = np.tile(rng.uniform(0.42, 0.75, episodes), P)
        m = np.minimum(env.W, 1.0); env.R, env.br = m * 0.11, m * 0.028
        env.rx = env.W / 2; env.rpx = env.rx.copy()
        side = np.tile(rng.choice([-1.0, 1.0], episodes), P)
        env.bx = env.W / 2 + side * env.W * 0.18; env.bvx = -side * env.W * 0.004
    ws = unpack(theta)
    for _ in range(max_steps):
        a = policy(ws, env.obs().reshape(P, episodes, N_IN)).reshape(P * episodes, N_OUT)
        env.step(a[:, 0], a[:, 1])
        if env.done.all():
            break
    return env.score.reshape(P, episodes).mean(1), env.steps.reshape(P, episodes).mean(1)


def heuristic_baseline(episodes=200, max_steps=7200, seed=999):
    """'Keep the racket under the ball' strategy, for comparison."""
    rng = np.random.default_rng(seed); env = Env(episodes, rng)
    for _ in range(max_steps):
        env.step(np.zeros(episodes), np.full(episodes, -0.6))
        if env.done.all(): break
    return env.score.mean()


def train(gens=300, pop=96, episodes=6, sigma=0.06, lr=0.03, seed=0):
    rng = np.random.default_rng(seed)
    D = n_params()
    theta = rng.normal(0, 0.15, D)
    m = np.zeros(D); v = np.zeros(D); b1, b2 = 0.9, 0.999
    best = (-1, theta.copy())
    t0 = time.time()
    for gen in range(1, gens + 1):
        max_steps = 3600 if gen < gens * 0.5 else 7200
        eps = rng.normal(0, 1, (pop // 2, D))
        eps = np.concatenate([eps, -eps])
        cand = theta + sigma * eps
        chunks = np.array_split(cand, POOL_N)
        res = POOL.starmap(rollout, [(c, episodes, max_steps, gen * 7919) for c in chunks])
        fit = np.concatenate([r[0] for r in res])
        ranks = np.empty(pop); ranks[fit.argsort()] = np.arange(pop)
        util = ranks / (pop - 1) - 0.5
        grad = (util @ eps) / (pop * sigma) - 0.005 * theta
        m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad ** 2
        theta = theta + lr * (m / (1 - b1 ** gen)) / (np.sqrt(v / (1 - b2 ** gen)) + 1e-8)
        if gen % 10 == 0 or gen == 1:
            cur, steps = rollout(theta[None], 48, 7200, seed=123456)
            if cur[0] > best[0]: best = (cur[0], theta.copy())
            print(f"gen {gen:4d}  pop mean {fit.mean():6.1f}  max {fit.max():6.1f}  "
                  f"| policy {cur[0]:6.1f} pts ({steps[0]/60:5.1f}s)  best {best[0]:6.1f}  [{time.time()-t0:5.0f}s]", flush=True)
    return best[1]


def export(theta, path, stats):
    W1, b1, W2, b2, W3, b3 = [w[0] for w in unpack(theta[None])]
    data = {
        "name": "padel-bounce-bot", "version": 1,
        "architecture": {"inputs": N_IN, "hidden": [N_H, N_H], "outputs": N_OUT, "activation": "tanh"},
        "inputs": ["ball_x/W*2-1", "ball_y/H*2-1", "ball_vx/H*40", "ball_vy/H*40",
                   "racket_x/W*2-1", "racket_y/H*2-1", "racket_vx/H*40", "racket_vy/H*40",
                   "(ball_x-racket_x)/W*2", "(ball_y-racket_y)/H*2", "wind/(W*0.007)",
                   "golden_ball(0|1)", "level*0.55", "W/H*2-1"],
        "outputs": ["pointer_x = ball_x + out0*W*0.5", "pointer_y = H*(0.5 + (out1+1)*0.25)"],
        "stats": stats,
        "weights": {k: np.round(w, 6).tolist() for k, w in
                    zip(["W1", "b1", "W2", "b2", "W3", "b3"], [W1, b1, W2, b2, W3, b3])},
    }
    with open(path, "w") as f:
        json.dump(data, f)


POOL, POOL_N = None, 4

if __name__ == "__main__":
    import multiprocessing as mp
    POOL = mp.Pool(POOL_N)
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    theta = train(gens)
    score, steps = rollout(theta[None], 400, 7200, seed=2024)
    base = heuristic_baseline()
    stats = {"eval_games": 400, "max_seconds": 120, "mean_score": round(float(score[0]), 1),
             "mean_seconds_alive": round(float(steps[0]) / 60, 1), "baseline_follow_ball_mean_score": round(float(base), 1)}
    print("FINAL", stats)
    import os
    export(theta, os.path.join(os.path.dirname(os.path.abspath(__file__)), "padel-bot-weights.json"), stats)
