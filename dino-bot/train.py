"""
Train a population of neural networks to play the endless runner in the fast
simulator, using a genetic algorithm.

    python train.py                 # train, saving the best brain to brains/best.npz
    python train.py --show          # same, but watch each generation play in a window
    python train.py --resume        # continue from brains/population.npz

Then watch the result with `python watch.py` or play real Chrome with `python chrome.py`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import brain
from runner_env import N_ACTIONS, OBS_SIZE, play


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--population", type=int, default=150, help="networks per generation")
    p.add_argument("--hidden", type=int, nargs="+", default=[16], help="hidden layer sizes, e.g. --hidden 16 8")
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--worlds", type=int, default=3, help="random worlds each network is scored on per generation")
    p.add_argument("--max-score", type=int, default=5000, help="end a game once this score is reached")
    p.add_argument("--elite", type=int, default=8)
    p.add_argument("--mutation-rate", type=float, default=0.3, help="fraction of weights mutated")
    p.add_argument("--mutation-sigma", type=float, default=0.2, help="size of mutations")
    p.add_argument("--crossover", type=float, default=0.0, help="chance a child mixes two parents")
    p.add_argument("--out", default="brains/best.npz", help="where to save the best brain")
    p.add_argument("--checkpoint", default="brains/population.npz")
    p.add_argument("--resume", action="store_true", help="continue from --checkpoint")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--show", action="store_true", help="watch training in a pygame window")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    sizes = [OBS_SIZE, *args.hidden, N_ACTIONS]
    start_gen = 0
    if args.resume and Path(args.checkpoint).exists():
        pop, sizes, start_gen = brain.load_population(args.checkpoint)
        print(f"Resumed {len(pop)} networks {sizes} at generation {start_gen}")
    else:
        pop = brain.random_population(sizes, args.population, rng)

    # A fixed set of unseen worlds to fairly compare "best so far" candidates.
    val_seeds = [10_000 + i for i in range(5)]
    best_val = -1.0
    if Path(args.out).exists() and args.resume:
        best_val = brain.load_brain(args.out)[2].get("validation_score", -1.0)

    viewer = None
    if args.show:
        from render import Viewer
        viewer = Viewer("dino-bot training")

    print(f"Network {sizes}, {brain.n_params(sizes)} weights, population {len(pop)}")
    for gen in range(start_gen, start_gen + args.generations):
        t0 = time.time()
        scores = []
        for w in range(args.worlds):
            policy = lambda obs, idx: brain.act(pop[idx], sizes, obs)
            on_frame = None
            if viewer and w == 0:
                on_frame = lambda world, dinos: viewer.draw(world, dinos, f"gen {gen}") or None
            scores.append(play(policy, len(pop), int(rng.integers(1 << 31)), args.max_score, on_frame))
            if viewer and viewer.closed:
                viewer = None
        fitness = np.mean(scores, axis=0)

        # Validate the top 3 on fixed worlds; keep the best brain ever seen.
        top = np.argsort(-fitness)[:3]
        val = np.mean([play(lambda o, i: brain.act(pop[top[i]], sizes, o), len(top), s, args.max_score)
                       for s in val_seeds], axis=0)
        champ = top[int(np.argmax(val))]
        saved = ""
        if val.max() > best_val:
            best_val = float(val.max())
            brain.save_brain(args.out, pop[champ], sizes, generation=gen, validation_score=best_val)
            saved = f"  -> saved {args.out}"

        print(f"gen {gen:4d} | best {fitness.max():7.0f} | median {np.median(fitness):6.0f} | "
              f"validation {val.max():7.0f} | {time.time() - t0:5.1f}s{saved}", flush=True)

        pop = brain.next_generation(pop, fitness, rng, args.elite, mutation_rate=args.mutation_rate,
                                    mutation_sigma=args.mutation_sigma, crossover=args.crossover)
        brain.save_population(args.checkpoint, pop, sizes, gen + 1)

        if best_val >= args.max_score:
            print(f"Best brain reaches the max score ({args.max_score}) on every validation world. Done!")
            break

    print(f"Best validation score: {best_val:.0f}  (brain: {args.out})")


if __name__ == "__main__":
    main()
