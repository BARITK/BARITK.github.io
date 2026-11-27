"""Hand-coded physics planner ("expert") used as the teacher for the neural network."""
import numpy as np
from train_padel_bot import G0

def expert_pointer(env, contact_margin=0.03, roof_margin=1.1, K=150):
    W, R, br = env.W, env.R, env.br
    tier = np.floor(env.score / 50)
    g = G0 * (1 + tier * 0.55)
    kick = 0.5 + tier * 0.22
    ymin, ymax = 0.5 + R, 1 - R * 0.6
    Yc = ymin + contact_margin
    # predict the ball (no racket) until it falls through the contact height
    x, y, vx, vy = env.bx.copy(), env.by.copy(), env.bvx.copy(), env.bvy.copy()
    k = np.full(env.n, K); xk = x.copy(); yk = y.copy(); vxk = vx.copy(); found = np.zeros(env.n, bool)
    t = env.t
    for j in range(1, K + 1):
        vy = vy + g
        w = env.wind * (0.7 + 0.3 * np.sin((t + j) * 0.12))
        vx = np.where(env.wind != 0, vx + (w - vx) * 0.05, vx)
        x = x + vx; y = y + vy
        lo = x < br; x = np.where(lo, br, x); vx = np.where(lo, np.abs(vx) * 0.95, vx)
        hi = x > W - br; x = np.where(hi, W - br, x); vx = np.where(hi, -np.abs(vx) * 0.95, vx)
        top = y < br; y = np.where(top, br, y); vy = np.where(top, np.abs(vy) * 0.6, vy)
        vx = vx * 0.999
        new = (~found) & (vy > 0) & (y >= Yc)
        k = np.where(new, j, k); xk = np.where(new, x, xk); yk = np.where(new, y, yk); vxk = np.where(new, vx, vxk)
        found |= new
        if found.all(): break
    xk = np.where(found, xk, x); yk = np.where(found, yk, Yc)
    # upward racket speed so the ball reaches the roof
    need = np.sqrt(2 * g * (yk - br)) * roof_margin
    U = np.maximum(0.003, need - 0.95 * np.sqrt(g) + 0.006)
    # horizontal racket speed that sends the ball back towards the middle
    last = env.lastHitX
    travel = np.where(np.isnan(last), 0.0, np.sign(xk - np.nan_to_num(last)))
    away = np.where(xk < W * 0.3, 1.0, np.where(xk > W * 0.7, -1.0, travel))
    T = 2 * np.sqrt(2 * yk / g) * 0.8
    vdes = (W / 2 - xk) / T * 0.8
    Vx = np.clip((vdes - vxk * 0.25 - away * 0.0035 * kick - env.wind * 1.6) / 0.45, -0.08, 0.08)
    m = k - 1
    dy = np.clip(yk + m * U, ymin, ymax)
    dx = np.clip(xk - np.minimum(m, 3) * Vx, R * 0.6, W - R * 0.6)
    px = env.rx + (dx - env.rx) / 0.75
    py = env.ry + (dy - env.ry) / 0.75
    return px, py
