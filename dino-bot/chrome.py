"""
Let a trained brain play the REAL dinosaur game in Chrome (chrome://dino),
or keep training (fine-tuning) it there.

    python chrome.py                       # watch brains/best.npz play in a Chrome window
    python chrome.py --train --tabs 6      # evolve further in real Chrome (headless tabs)

How it works: Playwright opens chrome://dino and injects a small script.  On
every animation frame the script reads the game's own state (dino height and
velocity, speed, next obstacles), computes the same features as the simulator,
runs the neural network (in JavaScript, so there is no lag) and presses jump /
duck exactly like the keyboard handlers do.  Python just hands over the
weights and collects scores.

One-time setup:  pip install playwright  &&  python -m playwright install chromium
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from playwright.sync_api import Page, sync_playwright

import brain

# Injected into the page.  Must stay in sync with runner_env.obstacle_features /
# dino_features and Dinos.act.
BOT_JS = r"""
() => {
  if (window.__dinoBot) return true;
  const r = Runner.getInstance();
  const WIDTH = 600, HEIGHT = 150, MAX_SPEED = 13, N_SEEN = 2;
  const NOOP = 0, JUMP = 1, DUCK = 2;
  const bot = window.__dinoBot = { genome: null, sizes: null, enabled: false, lastAction: 0 };

  bot.features = () => {
    const t = r.tRex, sp = r.currentSpeed;
    const f = [(t.groundYPos - t.yPos) / 100, t.jumpVelocity / 10, sp / MAX_SPEED];
    const ahead = r.horizon.obstacles
      .filter(o => o.xPos + o.width > t.xPos && !/collect/i.test(o.typeConfig.type))
      .sort((a, b) => a.xPos - b.xPos);
    for (let i = 0; i < N_SEEN; i++) {
      if (i < ahead.length) {
        const o = ahead[i], dx = o.xPos - t.xPos;
        f.push(dx / WIDTH, dx / sp / 60, o.yPos / HEIGHT, (o.yPos + o.typeConfig.height) / HEIGHT, o.width / 100);
      } else f.push(1.5, 1.5, 0, 0, 0);
    }
    return f;
  };

  bot.forward = (x) => {
    const s = bot.sizes, g = bot.genome;
    let i = 0;
    for (let l = 0; l < s.length - 1; l++) {
      const a = s[l], b = s[l + 1], y = new Array(b);
      for (let o = 0; o < b; o++) {
        let acc = g[i + a * b + o];
        for (let k = 0; k < a; k++) acc += x[k] * g[i + k * b + o];
        y[o] = l < s.length - 2 ? Math.tanh(acc) : acc;
      }
      i += a * b + b;
      x = y;
    }
    return x;
  };

  bot.apply = (a) => {
    const t = r.tRex;
    // Down key: duck on the ground, fast-fall in the air.
    if (a === DUCK) {
      if (t.jumping) { if (!t.speedDrop) t.setSpeedDrop(); }
      else if (!t.ducking) t.setDuck(true);
    } else {
      t.speedDrop = false;
      if (t.ducking) t.setDuck(false);
    }
    // Jump key: start a jump on the ground, letting go cuts the jump short.
    if (a === JUMP) { if (!t.jumping && !t.ducking) t.startJump(r.currentSpeed); }
    else if (t.jumping) t.endJump();
  };

  const originalUpdate = Object.getPrototypeOf(r).update;
  r.update = function (...args) {
    if (bot.enabled && bot.genome && this.activated && this.playing && !this.crashed) {
      const out = bot.forward(bot.features());
      const a = out.indexOf(Math.max(...out));
      bot.lastAction = a;
      bot.apply(a);
    }
    return originalUpdate.apply(this, args);
  };

  bot.load = (genome, sizes) => { bot.genome = genome; bot.sizes = sizes; bot.enabled = true; };
  bot.status = () => ({
    playing: r.playing, crashed: r.crashed,
    score: Math.ceil(r.distanceRan * 0.025), speed: r.currentSpeed,
  });
  bot.restart = () => { if (r.crashed) r.restart(); };
  // The game pauses when its tab is hidden; tabs used for training are never "hidden".
  document.addEventListener('visibilitychange', e => e.stopImmediatePropagation(), true);
  window.addEventListener('blur', e => e.stopImmediatePropagation(), true);
  return true;
}
"""


def open_game(page: Page) -> None:
    try:
        page.goto("chrome://dino")
    except Exception:
        pass  # Chrome reports the offline page as a navigation error; the game is still there.
    try:
        page.wait_for_function("typeof Runner !== 'undefined' && !!Runner.getInstance()", timeout=10_000)
    except Exception:
        raise SystemExit(
            "Couldn't find the dino game at chrome://dino.\n"
            "Use a full Chrome/Chromium (python -m playwright install chromium, or --browser chrome) and make\n"
            "sure the dino game isn't disabled by a policy (AllowDinosaurEasterEgg)."
        )
    page.evaluate(BOT_JS)


def load_genome(page: Page, genome: np.ndarray, sizes: list[int]) -> None:
    page.evaluate("([g, s]) => window.__dinoBot.load(g, s)", [genome.tolist(), sizes])


def start_or_restart(page: Page) -> None:
    st = page.evaluate("window.__dinoBot.status()")
    if st["crashed"]:
        page.evaluate("window.__dinoBot.restart()")
    elif not st["playing"]:
        page.keyboard.press("Space")       # the first game has to be started with a key press


def launch(p, args, headless: bool):
    kw = {"headless": headless}
    if args.executable:
        kw["executable_path"] = args.executable
    else:
        kw["channel"] = args.browser
    return p.chromium.launch(**kw)


# --------------------------------------------------------------------------- #
def play(args) -> None:
    genome, sizes, info = brain.load_brain(args.brain)
    print(f"Loaded {args.brain}: network {sizes}, {info}")
    with sync_playwright() as p:
        browser = launch(p, args, headless=args.headless)
        page = browser.new_page(viewport={"width": 900, "height": 500})
        open_game(page)
        load_genome(page, genome, sizes)
        games, scores = 0, []
        start_or_restart(page)
        try:
            while args.games == 0 or games < args.games:
                time.sleep(0.2)
                st = page.evaluate("window.__dinoBot.status()")
                if st["crashed"]:
                    games += 1
                    scores.append(st["score"])
                    print(f"game {games}: score {st['score']}   (best {max(scores)}, average {np.mean(scores):.0f})",
                          flush=True)
                    time.sleep(1.0)
                    start_or_restart(page)
        except KeyboardInterrupt:
            pass
        browser.close()


def train(args) -> None:
    rng = np.random.default_rng(args.seed)
    genome, sizes, info = brain.load_brain(args.brain)
    print(f"Fine-tuning {args.brain} in Chrome: population {args.population}, {args.tabs} tabs")
    pop = brain.population_from(genome, args.population, rng, sigma=args.mutation_sigma)
    best_score = -1

    with sync_playwright() as p:
        browser = launch(p, args, headless=not args.show)
        pages = []
        for _ in range(args.tabs):
            page = browser.new_page()
            open_game(page)
            pages.append(page)

        for gen in range(args.generations):
            t0 = time.time()
            fitness = np.zeros(len(pop))
            queue = [(i, e) for i in range(len(pop)) for e in range(args.episodes)]
            slots: list[tuple[int, float] | None] = [None] * len(pages)
            results: dict[int, list[int]] = {i: [] for i in range(len(pop))}

            while queue or any(slots):
                for k, page in enumerate(pages):
                    if slots[k] is None and queue:
                        i, _ = queue.pop(0)
                        load_genome(page, pop[i], sizes)
                        start_or_restart(page)
                        slots[k] = (i, time.time())
                        continue
                    if slots[k] is None:
                        continue
                    i, started = slots[k]
                    st = page.evaluate("window.__dinoBot.status()")
                    timed_out = time.time() - started > args.max_seconds
                    if st["crashed"] or timed_out:
                        results[i].append(st["score"])
                        if timed_out and not st["crashed"]:
                            # Stop this game by crashing it: simplest is to reload the page.
                            open_game(page)
                        slots[k] = None
                time.sleep(0.05)

            for i, sc in results.items():
                fitness[i] = np.mean(sc)
            champ = int(np.argmax(fitness))
            saved = ""
            if fitness[champ] > best_score:
                best_score = float(fitness[champ])
                brain.save_brain(args.out, pop[champ], sizes, generation=gen, chrome_score=best_score,
                                 fine_tuned_from=str(args.brain))
                saved = f"  -> saved {args.out}"
            print(f"gen {gen:3d} | best {fitness.max():6.0f} | median {np.median(fitness):6.0f} | "
                  f"{time.time() - t0:5.0f}s{saved}", flush=True)
            pop = brain.next_generation(pop, fitness, rng, elite=max(2, len(pop) // 10),
                                        mutation_rate=args.mutation_rate, mutation_sigma=args.mutation_sigma)
        browser.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("brain", nargs="?", default=None,
                   help="brain file (default: brains/best.npz, or brains/pretrained.npz if you haven't trained yet)")
    p.add_argument("--browser", default="chromium", help="Playwright channel: chromium (default), chrome, ...")
    p.add_argument("--executable", help="path to a Chrome/Chromium executable to use instead")
    p.add_argument("--headless", action="store_true", help="play without opening a window")
    p.add_argument("--games", type=int, default=0, help="games to play (0 = until Ctrl+C)")
    t = p.add_argument_group("training in Chrome (--train)")
    t.add_argument("--train", action="store_true", help="fine-tune the brain in real Chrome")
    t.add_argument("--tabs", type=int, default=4, help="games running in parallel")
    t.add_argument("--population", type=int, default=24)
    t.add_argument("--generations", type=int, default=10)
    t.add_argument("--episodes", type=int, default=1, help="games per network per generation")
    t.add_argument("--max-seconds", type=float, default=120, help="cut off a game after this long")
    t.add_argument("--mutation-rate", type=float, default=0.3)
    t.add_argument("--mutation-sigma", type=float, default=0.1)
    t.add_argument("--out", default="brains/chrome_best.npz")
    t.add_argument("--show", action="store_true", help="show the training browser window")
    t.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    args.brain = args.brain or brain.default_brain()
    train(args) if args.train else play(args)


if __name__ == "__main__":
    main()
