try:  # typing.Self requires python 3.11 or higher
    from typing import Self
except ImportError:
    try:  # maybe environment has `typing_extensions`
        from typing_extensions import Self
    except ImportError:  # fallback
        Self = 'Genome'  # type: ignore[assignment]


from typing import Any, Sequence

import numpy as np


class Genome:
    def __init__(self, gene_length: int = 16, *,
                 genes: Sequence[int] | None = None,
                 gene_set: set[Any] = None,
                 random_state=None):
        self._rng = np.random.default_rng(seed=random_state)

        if genes is not None:
            self.genes = np.asarray(genes)
            self.gene_length = len(genes)
        else:
            self.gene_length = gene_length
            self.genes = self._rng.integers(0, 2, size=gene_length)

        self.gene_set = gene_set or np.unique(genes)

    def crossover(self, other: Self, *, crossover_point: int = None) -> Self:
        assert self.gene_length == other.gene_length, (
            f'Genes are not of the same length, '
            f'{self.gene_length} != {other.gene_length}.'
        )

        if crossover_point is not None:
            assert isinstance(crossover_point, int), (
                'Crossover point must be an integer, '
                f'found type `{type(crossover_point)}`'
            )
            assert 0 < crossover_point < self.gene_length, (
                'Crossover point is not in bounds. '
                f'There are `{self.gene_length}` genes, '
                f'crossover point was found to be `{crossover_point}`'
            )
        else:
            crossover_point = self._rng.integers(1, self.gene_length)


if __name__ == '__main__':
    g = Genome(genes=['AA', 'BB', 'CC', 'DD'])
