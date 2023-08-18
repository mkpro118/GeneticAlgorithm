from typing import Optional, Iterable, Callable
import numpy as np

from genetic_algorithm.genome import Genome


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
                       count: Optional[int] = None) -> list[Genome]:
        # Calculate the fitness for each genome
        scores: Iterable = map(fitness_function, self.population)

        # int is a valid type for param `count`. Mypy expects a SupportsIndex
        scores = np.fromiter(scores, count=self.size)  # type: ignore

        # Perform roulette wheel selection based on fitness scores
        total_fitness: float = np.sum(scores)

        selection_probs = np.divide(scores, total_fitness, out=scores)

        def rand():
            return self._rng.choice(range(self.size), p=selection_probs)

        if count is not None:
            if not isinstance(count, int):
                raise TypeError(
                    f'`count` must be an int, found type `{type(count)}`.'
                )
            if count < 0:
                raise ValueError(
                    f'`count` must be greater than or equal to 0. found `{count=}`.'
                )
        else:
            count = self.size

        selected_parents = [self.population[rand()] for _ in range(self.size)]

        return selected_parents

    def evolve(self):
        selected_parents = self.select_parents()

        new_population = []

        while len(new_population) < self.size:
            parent1 = self._rng.choice(selected_parents)
            parent2 = self._rng.choice(selected_parents)

            child = parent1.crossover(parent2)
            child.mutate(self.mutation_rate, self.mutation_amount)

            new_population.append(child)

        self.population = new_population
