/*! Padel Bounce neural-network bot.
 * A 14-32-32-2 tanh MLP trained with evolution strategies (see train_padel_bot.py).
 * Usage: load this script on padel.html (or paste it into the browser console there),
 * then call PadelBot.start() — or use the "Watch AI play" button in the game.
 */
(function () {
  'use strict';
  const MODEL = __MODEL__;
  const w = MODEL.weights;

  function dense(x, Wm, b) {
    const out = new Array(b.length);
    for (let j = 0; j < b.length; j++) {
      let s = b[j];
      for (let i = 0; i < x.length; i++) s += x[i] * Wm[i][j];
      out[j] = Math.tanh(s);
    }
    return out;
  }
  function forward(x) { return dense(dense(dense(x, w.W1, w.b1), w.W2, w.b2), w.W3, w.b3); }

  // Reads the game's own globals (ball, racket, EV, S, W, H, pointer) from padel.html.
  function observe() {
    const level = Math.floor(S.score / 50);
    return [
      ball.x / W * 2 - 1, ball.y / H * 2 - 1, ball.vx / H * 40, ball.vy / H * 40,
      racket.x / W * 2 - 1, racket.y / H * 2 - 1, racket.vx / H * 40, racket.vy / H * 40,
      (ball.x - racket.x) / W * 2, (ball.y - racket.y) / H * 2,
      EV.wind / (W * 0.007), EV.type === 'gold' ? 1 : 0, level * 0.55, W / H * 2 - 1,
    ];
  }
  function act() {
    const [a0, a1] = forward(observe());
    pointer.grip = 0;
    pointer.x = ball.x + a0 * W * 0.5;
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
