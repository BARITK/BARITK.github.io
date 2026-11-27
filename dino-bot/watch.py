"""
Watch a trained brain play the simulator in a window.

    python watch.py                      # uses brains/best.npz
    python watch.py brains/other.npz
Keys: F = fast-forward, Esc = quit.
"""

from __future__ import annotations

import argparse

import numpy as np

import brain
from render import Viewer
from runner_env import play


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("brain", nargs="?", default=None,
                   help="brain file (default: brains/best.npz, or brains/pretrained.npz if you haven't trained yet)")
    p.add_argument("--games", type=int, default=0, help="number of games (0 = until you close the window)")
    args = p.parse_args()
    args.brain = args.brain or brain.default_brain()

    genome, sizes, info = brain.load_brain(args.brain)
    print(f"Loaded {args.brain}: network {sizes}, {info}")
    viewer = Viewer("dino-bot")
    g = 0
    while not viewer.closed and (args.games == 0 or g < args.games):
        g += 1
        score = play(lambda obs, idx: brain.act(genome[None], sizes, obs),
                     1, None, 10**9, lambda w, d: viewer.draw(w, d, f"game {g}"))
        if not viewer.closed:
            print(f"game {g}: score {int(score[0])}")
    viewer.close()


if __name__ == "__main__":
    main()
