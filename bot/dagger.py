"""Train the Padel Bounce network by imitation (DAgger) of the physics expert.

python3 dagger.py  -> writes padel-bot-weights.json
"""
import json, os, sys, time
import numpy as np
import train_padel_bot as T
from expert import expert_pointer

HID = [64, 64]
CAP = 2000          # stop a game once it reaches this score during data collection


class MLP:
    def __init__(self, sizes, rng):
        self.Ws = [rng.normal(0, 1 / np.sqrt(a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.bs = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.params()]; self.v = [np.zeros_like(p) for p in self.params()]; self.k = 0

    def params(self): return self.Ws + self.bs

    def forward(self, x, keep=False):
        hs = [x]
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            x = x @ W + b
            if i < len(self.Ws) - 1: x = np.tanh(x)
            hs.append(x)
        return (x, hs) if keep else x

    def train_step(self, x, y, lr):
        out, hs = self.forward(x, True)
        g = 2 * (out - y) / len(x)
        gW, gb = [None] * len(self.Ws), [None] * len(self.Ws)
        for i in reversed(range(len(self.Ws))):
            gW[i] = hs[i].T @ g; gb[i] = g.sum(0)
            if i: g = (g @ self.Ws[i].T) * (1 - hs[i] ** 2)
        self.k += 1
        for j, (p, gr) in enumerate(zip(self.params(), gW + gb)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * gr; self.v[j] = 0.999 * self.v[j] + 0.001 * gr * gr
            p -= lr * (self.m[j] / (1 - 0.9 ** self.k)) / (np.sqrt(self.v[j] / (1 - 0.999 ** self.k)) + 1e-8)
        return float(((out - y) ** 2).mean())


def expert_action(env):
    px, py = expert_pointer(env)
    R = env.R
    px = np.clip(px, R * 0.6, env.W - R * 0.6); py = np.clip(py, 0.5 + R, 1 - R * 0.6)
    return np.stack([(px / env.W - 0.5) / 0.6, (py - 0.5) / 0.25 - 1], 1)


def collect(net, n, seed, beta, max_steps=40000):
    """Play n games; act with expert (prob beta) or network; label every state with the expert."""
    rng = np.random.default_rng(seed); env = T.Env(n, rng)
    X, Y = [], []
    for s in range(max_steps):
        o = env.obs(); a_exp = expert_action(env)
        alive = ~env.done
        X.append(o[alive]); Y.append(a_exp[alive])
        a = a_exp if net is None else np.where(rng.random((n, 1)) < beta, a_exp, net.forward(o))
        env.step(a[:, 0], a[:, 1])
        env.done |= env.score >= CAP
        if env.done.all(): break
    return np.concatenate(X), np.concatenate(Y)


def evaluate(net, n=64, seed=777, cap=None, max_steps=60000):
    rng = np.random.default_rng(seed); env = T.Env(n, rng)
    for s in range(max_steps):
        a = net.forward(env.obs()); env.step(a[:, 0], a[:, 1])
        if cap: env.done |= env.score >= cap
        if env.done.all(): break
    return env.score, env.steps


def fit(net, X, Y, epochs, lr, bs=1024, rng=None):
    n = len(X)
    for e in range(epochs):
        idx = rng.permutation(n); tot = 0; c = 0
        for i in range(0, n, bs):
            j = idx[i:i + bs]; tot += net.train_step(X[j], Y[j], lr); c += 1
    return tot / c


def export(net, path, stats):
    data = {
        "name": "padel-bounce-bot", "version": 2,
        "architecture": {"inputs": T.N_IN, "hidden": HID, "outputs": 2, "hidden_activation": "tanh", "output_activation": "linear"},
        "inputs": ["ball_x/W*2-1", "ball_y/H*2-1", "ball_vx/H*40", "ball_vy/H*40",
                   "racket_x/W*2-1", "racket_y/H*2-1", "racket_vx/H*40", "racket_vy/H*40",
                   "(ball_x-racket_x)/W*2", "(ball_y-racket_y)/H*2", "wind/(W*0.007)",
                   "golden_ball(0|1)", "level*0.55", "W/H*2-1", "(racket_x-last_hit_x)/W*2 (0 before first hit)", "roof_bonus_armed(0|1)"],
        "outputs": ["pointer_x = W*(0.5 + out0*0.6)", "pointer_y = H*(0.5 + (out1+1)*0.25)"],
        "training": "Imitation learning (DAgger) of a physics-planner teacher in a simulator ported from padel.html",
        "stats": stats,
        "weights": {"layers": [{"W": np.round(W, 6).tolist(), "b": np.round(b, 6).tolist()} for W, b in zip(net.Ws, net.bs)]},
    }
    json.dump(data, open(path, "w"))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    net = MLP([T.N_IN] + HID + [2], rng)
    t0 = time.time()
    X, Y = collect(None, 96, seed=1, beta=1)
    print(f"expert data {len(X)} states  [{time.time()-t0:.0f}s]", flush=True)
    loss = fit(net, X, Y, 8, 2e-3, rng=rng)
    best = (-1, None)
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    for it in range(1, iters + 1):
        beta = max(0.0, 0.5 - 0.1 * it)
        Xn, Yn = collect(net, 96, seed=100 + it, beta=beta)
        X = np.concatenate([X, Xn]); Y = np.concatenate([Y, Yn])
        if len(X) > 3_000_000:
            keep = rng.choice(len(X), 3_000_000, replace=False); X, Y = X[keep], Y[keep]
        lr = 2e-3 if it < iters * 0.6 else 6e-4
        loss = fit(net, X, Y, 4, lr, rng=rng)
        sc, st = evaluate(net, 64, seed=777, cap=CAP)
        print(f"iter {it:2d}  data {len(X):8d}  loss {loss:.5f}  net: mean {sc.mean():6.0f}  median {np.median(sc):6.0f}  "
              f"min {sc.min():5.0f}  reached1500 {(sc>=1500).sum():2d}/64  [{time.time()-t0:.0f}s]", flush=True)
        if np.median(sc) + (sc >= 1500).mean() * 1000 > best[0]:
            best = (np.median(sc) + (sc >= 1500).mean() * 1000, [W.copy() for W in net.Ws], [b.copy() for b in net.bs])
    net.Ws, net.bs = best[1], best[2]
    sc, st = evaluate(net, 200, seed=4242, cap=None, max_steps=120000)
    stats = {"eval_games": 200, "mean_score": round(float(sc.mean()), 1), "median_score": float(np.median(sc)),
             "min_score": float(sc.min()), "max_score": float(sc.max()), "games_reaching_1500": int((sc >= 1500).sum()),
             "mean_minutes_alive": round(float(st.mean()) / 3600, 1)}
    print("FINAL", json.dumps(stats), flush=True)
    export(net, os.path.join(os.path.dirname(os.path.abspath(__file__)), "padel-bot-weights.json"), stats)
