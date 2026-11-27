# dino-bot

A neural network that learns to play endless-runner games like Chrome's dinosaur
game (`chrome://dino`). You can train it on your own PC in a couple of minutes,
then watch it play the **real** game in Chrome.

It uses **neuroevolution**: a population of small neural networks plays, the
best ones survive, and mutated copies of them form the next generation. There's
no deep-learning framework involved, just numpy.

```
observation (13 numbers) ──► 16 hidden neurons (tanh) ──► 3 outputs: run / jump / duck
```

## Setup

Python 3.9+:

```bash
cd dino-bot
pip install -r requirements.txt
python -m playwright install chromium      # only needed for playing in Chrome
```

## 1. Train (fast simulator)

```bash
python train.py            # prints progress, saves brains/best.npz
python train.py --show     # same, and you watch every generation play (press F to fast-forward)
```

`train.py` runs a headless simulator that uses the same physics as Chrome's
game: gravity, jump speed, how fast it speeds up, obstacle sizes and gaps,
pterodactyl heights and collision boxes. All 150 dinos in a generation run in
the same world at once, with vectorised numpy, so a generation takes about a
second at the start. On a normal laptop it usually beats score 2000 on every
test world within 15–30 generations, which is 1–3 minutes.

Useful options:

| option | default | meaning |
|---|---|---|
| `--population` | 150 | networks per generation |
| `--hidden 16 8` | `16` | hidden layer sizes |
| `--generations` | 100 | stops early once the best brain hits `--max-score` everywhere |
| `--max-score` | 5000 | games end at this score, so a perfect bot doesn't run forever |
| `--worlds` | 3 | random worlds each network is scored on per generation |
| `--mutation-rate` / `--mutation-sigma` | 0.3 / 0.2 | how much children differ from parents |
| `--resume` | | continue from `brains/population.npz` |

```bash
python watch.py            # watch your best brain play the simulator
```

## 2. Play the real Chrome dino game

```bash
python chrome.py                    # opens a Chrome window and plays until Ctrl+C
python chrome.py --browser chrome   # use your installed Google Chrome instead of Playwright's Chromium
```

If you haven't trained yet, it uses the bundled `brains/pretrained.npz`. In
testing, that brain ran the real game past 3000 at top speed without crashing.

How it works: Playwright opens `chrome://dino` and injects a small script. On
every animation frame the script reads the game's state (dino height and
velocity, speed, the next two obstacles), turns it into exactly the same 13
numbers the simulator gives the network, and runs the network in JavaScript,
so there's no lag. It then presses jump or duck the same way the keyboard
handlers do. The bot reads the game's internal state rather than the pixels on
screen.

## 3. (Optional) Keep training inside real Chrome

```bash
python chrome.py --train --tabs 6 --population 24 --generations 10
```

This starts from your trained brain and keeps evolving it, with several real
games running in parallel headless tabs. It runs in real time, so it's much
slower than the simulator; use it to polish a brain, not to train one from
scratch. The best brain is saved to `brains/chrome_best.npz`, and you can play
it with `python chrome.py brains/chrome_best.npz`. Add `--show` to watch the tabs.

## Files

| file | what it is |
|---|---|
| `brain.py` | neural network (batched forward pass), genetic algorithm, save/load. Game-agnostic. |
| `runner_env.py` | the dino-style simulator, the observation features, and `play()` |
| `train.py` | training loop |
| `watch.py`, `render.py` | pygame viewer |
| `chrome.py` | plays and fine-tunes in real Chrome via Playwright |

## Training it on other games

`brain.py` doesn't know anything about dinosaurs. To teach it another game you need:

1. **An observation**: a fixed-length list of numbers, each roughly in −1…1,
   describing what matters (positions, distances, speeds). Precomputed
   "time until X arrives" features help a lot. Adding one sped up learning
   here several times over.
2. **A small set of actions** for the network to choose from (the size of the output layer).
3. **A score** to maximise (survival time, points, …).

Then the loop is the same as in `train.py`:

```python
import numpy as np, brain
sizes = [OBS_SIZE, 16, N_ACTIONS]
pop = brain.random_population(sizes, 150, rng)
for gen in range(100):
    fitness = np.array([run_my_game(lambda obs: brain.act(pop[i:i+1], sizes, obs[None])[0])
                        for i in range(len(pop))])
    pop = brain.next_generation(pop, fitness, rng)
```

For a game you can't simulate, do what `chrome.py` does: read the game's
state (from its JavaScript, memory, or screen pixels), compute the same
observation, and send key presses.

## Tips

* Learning stuck at a low score? Try more `--worlds` (less luck involved) or a
  bigger `--population`.
* Crossover (`--crossover 0.5`) is off by default. Mixing two unrelated
  networks usually broke both, and mutation alone learned about 10× faster in testing.
* If `chrome://dino` doesn't load the game, your Chrome may have it disabled by
  policy (`AllowDinosaurEasterEgg`). Use Playwright's Chromium (the default) instead.
