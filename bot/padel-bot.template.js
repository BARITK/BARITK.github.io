/*! Padel Bounce neural-network bot.
 * A 16-64-64-2 MLP (tanh hidden layers, linear output) trained by imitation learning (DAgger)
 * against a physics-planner teacher, in a simulator ported from the game (see dagger.py).
 * Usage: load this script on padel.html (or paste it into the browser console there),
 * then call PadelBot.start() — or use the "Watch AI play" button in the game.
 */
(function () {
  'use strict';
  const MODEL = __MODEL__;
  const layers = MODEL.weights.layers;

  function forward(x) {
    for (let l = 0; l < layers.length; l++) {
      const Wm = layers[l].W, b = layers[l].b, out = new Array(b.length);
      for (let j = 0; j < b.length; j++) {
        let s = b[j];
        for (let i = 0; i < x.length; i++) s += x[i] * Wm[i][j];
        out[j] = l < layers.length - 1 ? Math.tanh(s) : s;
      }
      x = out;
    }
    return x;
  }

  // Reads the game's own globals (ball, racket, EV, S, W, H, pointer) from padel.html.
  function observe() {
    const level = Math.floor(S.score / 50);
    return [
      ball.x / W * 2 - 1, ball.y / H * 2 - 1, ball.vx / H * 40, ball.vy / H * 40,
      racket.x / W * 2 - 1, racket.y / H * 2 - 1, racket.vx / H * 40, racket.vy / H * 40,
      (ball.x - racket.x) / W * 2, (ball.y - racket.y) / H * 2,
      EV.wind / (W * 0.007), EV.type === 'gold' ? 1 : 0, level * 0.55, W / H * 2 - 1,
      S.lastHitX === null ? 0 : (racket.x - S.lastHitX) / W * 2,
      S.ceilingArmed ? 1 : 0,
    ];
  }
  function act() {
    const [a0, a1] = forward(observe());
    pointer.grip = 0;
    pointer.x = W * (0.5 + a0 * 0.6);
    pointer.y = H * (0.5 + (a1 + 1) * 0.25);
  }

  const PadelBot = {
    model: MODEL, forward, observe, enabled: false,
    start() { this.enabled = true; if (!S.running && typeof begin === 'function') begin(true); },
    stop() { this.enabled = false; },
  };
  if (typeof window.tick === 'function' && !window.tick.__bot) {
    const orig = window.tick;
    window.tick = function () { if (PadelBot.enabled && S.running) act(); return orig.apply(this, arguments); };
    window.tick.__bot = true;
  }
  window.PadelBot = PadelBot;
})();
