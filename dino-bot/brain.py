"""
Neural network + genetic algorithm ("neuroevolution"), written with plain numpy.

Nothing in this file knows about the dino game, so you can reuse it for any game
that can give you a vector of numbers (the observation) and wants back one of a
few discrete actions.

A *genome* is just every weight and bias of a small fully-connected network,
flattened into one long vector.  A *population* is a 2-D array of genomes,
shape (population_size, n_params), which lets us run the forward pass for every
individual at once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def n_params(sizes: list[int]) -> int:
    """Number of weights + biases in an MLP with the given layer sizes."""
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def random_population(sizes: list[int], n: int, rng: np.random.Generator) -> np.ndarray:
    """Xavier-ish random init for `n` networks, returned as a (n, n_params) array."""
    parts = []
    for a, b in zip(sizes, sizes[1:]):
        parts.append(rng.normal(0.0, 1.0 / np.sqrt(a), size=(n, a * b)))
        parts.append(np.zeros((n, b)))
    return np.concatenate(parts, axis=1)


def forward(genomes: np.ndarray, sizes: list[int], x: np.ndarray) -> np.ndarray:
    """
    Run every network in `genomes` (n, P) on its own input row of `x` (n, in).
    Hidden layers use tanh, output layer is linear.  Returns (n, out).
    """
    i = 0
    n_layers = len(sizes) - 1
    for layer, (a, b) in enumerate(zip(sizes, sizes[1:])):
        W = genomes[:, i : i + a * b].reshape(-1, a, b)
        i += a * b
        bias = genomes[:, i : i + b]
        i += b
        x = np.einsum("ni,nio->no", x, W) + bias
        if layer < n_layers - 1:
            x = np.tanh(x)
    return x


def act(genomes: np.ndarray, sizes: list[int], x: np.ndarray) -> np.ndarray:
    """Pick the action with the highest output for each network."""
    return np.argmax(forward(genomes, sizes, x), axis=1)


# --------------------------------------------------------------------------- #
# Genetic algorithm
# --------------------------------------------------------------------------- #
def next_generation(
    genomes: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator,
    elite: int = 8,
    tournament: int = 4,
    mutation_rate: float = 0.3,
    mutation_sigma: float = 0.2,
    crossover: float = 0.0,
) -> np.ndarray:
    """
    Breed a new population:
      * the `elite` best genomes are copied unchanged,
      * every other child is a copy of a tournament winner (with probability
        `crossover`, mixed gene-by-gene with a second winner), then gaussian
        noise is added to a random `mutation_rate` fraction of its genes.

    Crossover is off by default: two unrelated networks usually encode the
    same skill with different neurons, so mixing them tends to break both.
    Mutation alone learned the dino game ~10x faster in testing.
    """
    n, P = genomes.shape
    order = np.argsort(-fitness)
    elites = genomes[order[:elite]]

    k = n - elite

    def tournament_pick(count: int) -> np.ndarray:
        contenders = rng.integers(0, n, size=(count, tournament))
        winners = contenders[np.arange(count), np.argmax(fitness[contenders], axis=1)]
        return genomes[winners]

    children = tournament_pick(k)
    if crossover > 0:
        dads = tournament_pick(k)
        mix = (rng.random((k, 1)) < crossover) & (rng.random((k, P)) < 0.5)
        children = np.where(mix, dads, children)
    mutate = rng.random((k, P)) < mutation_rate
    children = children + mutate * rng.normal(0.0, mutation_sigma, size=(k, P))
    return np.concatenate([elites, children], axis=0)


def population_from(genome: np.ndarray, n: int, rng: np.random.Generator, sigma: float = 0.1) -> np.ndarray:
    """A population of mutated copies of one genome (the first copy is exact)."""
    pop = np.repeat(genome[None, :], n, axis=0)
    pop[1:] += rng.normal(0.0, sigma, size=pop[1:].shape)
    return pop


# --------------------------------------------------------------------------- #
# Saving / loading
# --------------------------------------------------------------------------- #
def save_brain(path: str | Path, genome: np.ndarray, sizes: list[int], **info) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, genome=genome, sizes=np.array(sizes), info=json.dumps(info))


def default_brain() -> str:
    """brains/best.npz if you've trained one, otherwise the bundled pretrained brain."""
    here = Path(__file__).resolve().parent
    for name in ("brains/best.npz", "brains/pretrained.npz"):
        if Path(name).exists():
            return name
        if (here / name).exists():
            return str(here / name)
    raise SystemExit("No brain found. Train one first:  python train.py")


def load_brain(path: str | Path) -> tuple[np.ndarray, list[int], dict]:
    data = np.load(path, allow_pickle=False)
    info = json.loads(str(data["info"])) if "info" in data else {}
    return data["genome"], [int(s) for s in data["sizes"]], info


def save_population(path: str | Path, genomes: np.ndarray, sizes: list[int], generation: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, genomes=genomes, sizes=np.array(sizes), generation=generation)


def load_population(path: str | Path) -> tuple[np.ndarray, list[int], int]:
    data = np.load(path)
    return data["genomes"], [int(s) for s in data["sizes"]], int(data["generation"])
