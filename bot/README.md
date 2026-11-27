# Padel Bounce AI bot

A neural network that plays **Padel Bounce** (`padel.html`).

## Results
- Simulator, 200 games: median **2,361** points, mean 2,403. **197 of 200 games reached 1,500.**
- Real game in a browser (4 games at phone sizes): 1,798 · 1,744 · 1,742 · 2,027.

## Files
| File | What it is |
|---|---|
| `padel-bot.js` | The bot, ready to run, with the trained weights embedded. |
| `padel-bot-weights.json` | The trained weights on their own, with input/output definitions. |
| `train_padel_bot.py` | Game simulator, a port of the game's 60 Hz physics (plus an earlier evolution-strategies trainer). |
| `expert.py` | Physics-planner teacher. |
| `dagger.py` | Imitation-learning (DAgger) training that produced the weights. |
| `build_bot_js.py`, `padel-bot.template.js` | Rebuild `padel-bot.js` from the weights. |

## Run it
- **In the game:** put `padel-bot.js` in a `bot/` folder next to `padel.html`. A **Watch AI play** button then appears on the start screen.
- **Manually:** open `padel.html`, paste `padel-bot.js` into the browser console, then run `PadelBot.start()`.
- **On Android or BlueStacks:** open the hosted `padel.html` in Chrome and tap **Watch AI play**.

The bot reads the game's internal state (ball, racket, wind, level). It works only with this game.

## Model
- Network: 16 inputs → 64 → 64 (tanh) → 2 outputs (linear), the pointer x and y.
- Training: DAgger imitation of a planner that strikes the ball upward to hit the roof (+5) on almost every bounce and steers it back to the middle.
- Retrain: `python3 dagger.py 12` (numpy only, about 8 minutes on 4 cores), then `python3 build_bot_js.py`.
