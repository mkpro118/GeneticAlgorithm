from typing import Sequence, Optional, Iterable

import numpy as np


class Genome:
    def __init__(self, gene_length: int = 16, *,
                 genes: Optional[Sequence[int]] = None,
                 gene_set: Optional[Iterable] = None,
                 gene_range: Optional[tuple[float, float]] = None,
                 random_state: Optional[int] = None):
        self._rng = np.random.default_rng(seed=random_state)

        if genes is not None:
            self.genes = np.asarray(genes)
            self.gene_length = len(genes)
        else:
            self.gene_length = gene_length
            self.genes = self._rng.integers(0, 2, size=gene_length)

        self._check_gene_set(gene_set, gene_range)

    def _check_gene_set(self, gene_set: Optional[Iterable],
                        gene_range: Optional[tuple[float, float]]):
        if not np.issubdtype(self.genes.dtype, np.floating):
            if gene_range is None:
                self.gene_set = tuple(gene_set or set(self.genes))
                self._continuous = False
                return
            else:
                try:
                    self.genes = self.genes.astype(int)
                except TypeError:
                    raise ValueError(
                        '`gene_range` was provided, but data could not be cast '
                        'into floats. `gene_range` should only be used with '
                        'floating-point/real-valued data. For discrete data, '
                        'use `gene_set`.'
                    )

        if gene_range is None:
            raise ValueError(
                'Real values detected, use `gene_range` to set the range of real values'
            )

        if (_ := len(gene_range)) != 2:
            raise ValueError(
                '`gene_range` must be a 2-tuple of the lower and upper bound '
                f'of the real values. Found `len(gene_range)={_}`'
            )

        if not all(isinstance(x, (int, float)) for x in gene_range):
            raise ValueError(
                '`gene_range` must be a 2-tuple of floats, found '
                f'({type(gene_range[0])}, {type(gene_range[1])})'
            )

        self.gene_range = tuple(map(float, sorted(gene_range)))
        self._continuous = True

        self.genes = np.clip(self.genes, *self.gene_range)

    def crossover(self, other: 'Genome', *,
                  crossover_point: int | None = None) -> 'Genome':
        if self.gene_length != other.gene_length:
            raise ValueError(
                f'Genes are not of the same length, '
                f'{self.gene_length} != {other.gene_length}.'
            )

        if crossover_point is not None:
            if not isinstance(crossover_point, int):
                raise TypeError(
                    'crossover_point must be an integer, '
                    f'found type `{type(crossover_point)}`'
                )
            if not (0 < crossover_point < self.gene_length):
                raise ValueError(
                    'Crossover point is not in bounds. '
                    f'There are `{self.gene_length}` genes, but '
                    f'crossover point was found to be `{crossover_point}`'
                )
        else:
            crossover_point = self._rng.integers(1, self.gene_length)

        first_half = self.genes[:crossover_point]
        second_half = other.genes[crossover_point:]
        child_genes = np.concatenate((first_half, second_half))

        return Genome(genes=child_genes, gene_set=self.gene_set)

    def mutate(self, mutation_rate: float = 0.01, *,
               mutation_amount: Optional[float] = None,
               inplace: bool = True) -> 'Genome':
        if not isinstance(mutation_rate, float):
            raise TypeError(
                f'mutation_rate must be a float, found type `{type(mutation_rate)}`'
            )

        if not 0. <= mutation_rate <= 1.:
            raise ValueError(
                '`mutation_rate` must be a probability value between 0 and 1 '
                f'(both inclusive), found `{mutation_rate = }`.'
            )

        mask = self._rng.random(size=self.gene_length) < mutation_rate

        if self._continuous:
            if mutation_amount is None:
                raise ValueError(
                    '`mutation_amount` must be provided for mutation of real valued genes.'
                )

            if not isinstance(mutation_amount, (int, float)):
                raise TypeError(
                    f'`mutation_amount` must be a float, found type `{type(mutation_amount)}`'
                )
            mutated = self._mutate_continuous(mask, mutation_amount)
        else:
            mutated = self._mutate_discrete(mask)

        if inplace:
            self.genes[mask] = mutated
            return self
        else:
            child_genes = self.genes.copy()
            child_genes[mask] = mutated
            return Genome(genes=child_genes, gene_set=self.gene_set)

    def _mutate_continuous(self, mask: np.ndarray, amount: float) -> np.ndarray:
        return self.genes[mask] + self._rng.uniform(-amount, amount, size=np.sum(mask))

    def _mutate_discrete(self, mask: np.ndarray) -> np.ndarray:
        return self._rng.choice(self.gene_set, size=np.sum(mask))

    def __str__(self):
        return f'Genome(genes={self.genes})'

    def __repr__(self):
        return (
            f'Genome(length={self.gene_length}, gene_set={self.gene_set}, '
            f'genes={self.genes})'
        )


if __name__ == '__main__':
    g = Genome(genes=np.arange(1, 5), gene_range=(0, 6))
    print(g)
    g.mutate(mutation_rate=0.6, mutation_amount=0.05)
    print(g)
