from typing import Callable, Iterable, Optional
import numpy as np

from genetic_algorithm.genome import Genome
from genetic_algorithm.selection import selection_methods


class Population:
    def __init__(self, size: int,
                 gene_length: int = 16,
                 gene_set: Optional[Iterable] = None,
                 gene_range: Optional[tuple[float, float]] = None,
                 mutation_rate: float = 0.01,
                 mutation_amount: float = 0.1,
                 random_state: Optional[int] = None):
        self.size = size
        self.gene_length = gene_length
        self.gene_set = gene_set
        self.gene_range = gene_range
        self.mutation_rate = mutation_rate
        self.mutation_amount = mutation_amount
        self._rng = np.random.default_rng(seed=random_state)

        if random_state is not None:
            if not isinstance(random_state, int):
                raise TypeError(
                    f'`random_state` must be an integer, found type `{type(random_state)}`'
                )

            self._random_state = lambda x: random_state + x
        else:
            self._random_state = lambda _: None

    def initialize(self):
        self.population = [Genome(gene_length=self.gene_length,
                                  gene_set=self.gene_set,
                                  gene_range=self.gene_range,
                                  random_state=self._random_state(_)) for _ in range(self.size)]

    def select_parents(self, fitness_function: Callable[[Genome], float],
                       method: str = 'roulette',
                       count: Optional[int] = None,
                       random_state: Optional[int] = None,
                       **kwargs) -> list[Genome]:
        # Validate fitness_function
        if not callable(fitness_function):
            raise TypeError(
                '`fitness_function` must be a callable that returns a fitness score, '
                f'found type {type(fitness_function)}.'
            )

        # Calculate the fitness for each genome
        scores: Iterable = map(fitness_function, self.population)

        # int is a valid type for param `count`. Mypy expects a SupportsIndex
        scores = np.fromiter(scores, count=self.size)  # type: ignore

        try:
            return selection_methods[method](self.population,
                                             scores,
                                             count=count,
                                             random_state=random_state,
                                             **kwargs)
        except KeyError:
            raise ValueError(
                f'`{method=}` is not a known selection method'
            )

    def evolve(self, fitness_function: Callable[[Genome], float],
               selection_method: str = 'roulette',
               selection_count: Optional[int] = None,
               crossover_method: str = 'one_point',
               random_state: Optional[int] = None, **kwargs):

        rate = kwargs.get('mutation_rate', self.mutation_rate)
        amount = kwargs.get('mutation_amount', self.mutation_amount)

        selected_parents = self.select_parents(fitness_function,
                                               method=selection_method,
                                               count=selection_count,
                                               random_state=random_state)

        new_population: list[Genome] = []

        while len(new_population) < self.size:
            parent1 = self._rng.choice(selected_parents)
            parent2 = self._rng.choice(selected_parents)

            child = parent1.crossover(parent2,
                                      method=crossover_method,
                                      **kwargs)

            child.mutate(rate, amount)

            new_population.append(child)

        self.population = new_population
