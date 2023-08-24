from typing import Sequence, Optional

import numpy as np

from genetic_algorithm.genome import Genome
from genetic_algorithm.utils import is_int, kwargs_proxy


def _check_count(count):
    # Validate count
    if not is_int(count):
        raise TypeError(
            f'`count` must be an int, found type `{type(count)}`.'
        )
    if count < 0:
        raise ValueError(
            f'`count` must be greater than or equal to 0. found `{count=}`.'
        )


@kwargs_proxy
def boltzmann(population: Sequence[Genome],
              fitness_scores: Sequence[float],
              temperature: float = 1.,
              count: Optional[int] = None,
              random_state: Optional[int] = None) -> list[Genome]:
    size = len(population)

    if count is not None:
        _check_count(count)
    else:
        count = size

    scores = np.asarray(fitness_scores)

    rng = np.random.default_rng(seed=random_state)

    exp_scores = np.exp(scores / temperature)
    selection_probs = exp_scores / np.sum(exp_scores)

    selected_indices = rng.choice(size, size=count, p=selection_probs)
    selected_parents = [population[i] for i in selected_indices]

    return selected_parents


@kwargs_proxy
def ranked(population: Sequence[Genome],
           fitness_scores: Sequence[float],
           count: Optional[int] = None,
           random_state: Optional[int] = None) -> list[Genome]:
    size = len(population)

    if count is not None:
        _check_count(count)
    else:
        count = size

    scores = np.asarray(fitness_scores)

    rng = np.random.default_rng(seed=random_state)

    ranked_idxs = np.argsort(scores)

    selection_probs = np.arange(1, size + 1)
    selection_probs = selection_probs / np.sum(selection_probs)

    selected_idxs = rng.choice(ranked_idxs, size=count, p=selection_probs)

    selected_parents = [population[i] for i in selected_idxs]

    return selected_parents


@kwargs_proxy
def roulette(population: Sequence[Genome],
             fitness_scores: Sequence[float],
             count: Optional[int] = None,
             random_state: Optional[int] = None) -> list[Genome]:

    size = len(population)

    if count is not None:
        _check_count(count)
    else:
        count = size

    scores = np.asarray(fitness_scores)

    # Perform roulette wheel selection based on fitness scores
    total_fitness: float = np.sum(scores)

    selection_probs = np.divide(scores, total_fitness)

    rng = np.random.default_rng(seed=random_state)

    def rand():
        return rng.choice(range(size), p=selection_probs)

    selected_parents = [population[rand()] for _ in range(count)]

    return selected_parents


@kwargs_proxy
def stochastic_universal_sampling(population: Sequence[Genome],
                                  fitness_scores: Sequence[float],
                                  count: Optional[int] = None,
                                  random_state: Optional[int] = None) -> list[Genome]:
    size = len(population)

    if count is not None:
        _check_count(count)
    else:
        count = size

    scores = np.asarray(fitness_scores)

    rng = np.random.default_rng(seed=random_state)

    total_fitness: float = np.sum(scores)
    selection_probs = scores / total_fitness

    step_size = total_fitness / count

    current_point = rng.uniform(0, step_size)

    selected_parents = []

    for _ in range(count):
        while current_point > selection_probs[0]:
            current_point -= step_size
            selection_probs = selection_probs[1:]

        selected_index = np.argmax(selection_probs >= current_point)

        # selected_index is an integer, mypy doesn't handle conversions
        # between numpy integers and python integers
        selected_parents.append(population[selected_index])  # type: ignore

        current_point += step_size

    return selected_parents


@kwargs_proxy
def tournament(population: Sequence[Genome],
               fitness_scores: Sequence[float],
               n_players: int = 2,
               count: Optional[int] = None,
               random_state: Optional[int] = None) -> list[Genome]:
    size = len(population)

    if count is not None:
        _check_count(count)
    else:
        count = size

    scores = np.asarray(fitness_scores)

    rng = np.random.default_rng(seed=random_state)

    selected_parents = []

    for i in range(count):
        player_idxs = rng.choice(size, size=n_players, replace=False)
        best_player_idx = np.argmax(scores[player_idxs])
        winner_index = player_idxs[best_player_idx]
        selected_parents.append(population[winner_index])

    return selected_parents


selection_methods = {
    'boltzmann': boltzmann,
    'ranked': ranked,
    'roulette': roulette,
    'stochastic_universal_sampling': stochastic_universal_sampling,
    'sus': stochastic_universal_sampling,
    'tournament': tournament,
}
