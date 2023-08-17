from typing import Optional, Iterable, Sequence

from numpy.typing import NDArray
import numpy as np


class Genome:
    """
    Represents a genome with gene sequences that can undergo mutation and crossover.

    Attributes:
        genes (np.ndarray): Sequence of gene values.
        gene_length (int): Length of the gene sequence.
        gene_set (tuple): Set of possible gene values. Only defined for discrete data
        gene_range (tuple): Range of real-valued gene values. Only defined for continuous data

    Methods:
        crossover:
            Perform crossover with another genome to create a new genome.

        mutate:
            Mutate the genome by altering gene values based on mutation rate and amount.

    """

    def __init__(self, gene_length: int = 16, *,
                 genes: Optional[Sequence | NDArray] = None,
                 gene_set: Optional[Iterable] = None,
                 gene_range: Optional[tuple[float, float]] = None,
                 random_state: Optional[int] = None):
        """
        Initialize a Genome instance.

        Args:
            gene_length (int): The length of the gene sequence (default is 16).
            genes (Optional[NDArray[int]]): A sequence of gene values (optional).
            gene_set (Optional[Iterable]): A set of possible gene values (optional).
            gene_range (Optional[tuple[float, float]]): The range of real-valued gene values (optional).
            random_state (Optional[int]): Seed for the random number generator (optional).
        """
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
        """
        Validate and set the gene_set and gene_range attributes.

        Args:
            gene_set (Optional[Iterable]): A set of possible gene values (optional).
            gene_range (Optional[tuple[float, float]]): The range of real-valued gene values (optional).
        """
        # If the gene values are not real values
        if not np.issubdtype(self.genes.dtype, np.floating):
            # Define a gene_set for discrete values
            if gene_range is None:
                self.gene_set = tuple(gene_set or set(self.genes))
                self._continuous = False
                return
            else:
                # If `gene_range` is provided, but data is not real-valued
                # We try to convert the data into floats
                try:
                    self.genes = self.genes.astype(float)
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

        # Set gene_range and continuous flag for real-valued genes
        self.gene_range = tuple(float(x) for x in gene_range)
        self._continuous = True

        # Clip gene values to fit within the defined gene_range
        self.genes = np.clip(
            self.genes, self.gene_range[0], self.gene_range[1])

    def crossover(self, other: 'Genome', *,
                  crossover_point: int | None = None) -> 'Genome':
        """
        Perform crossover with another genome to create a new genome.

        Args:
            other (Genome): The other genome to perform crossover with.
            crossover_point (int | None): The index to perform crossover at, or None for random (default).

        Returns:
            Genome: A new genome resulting from crossover.
        """
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
            # Choose a random crossover point if one is not provided
            crossover_point = self._rng.integers(1, self.gene_length)

        # Perform crossover by combining genes from both parents at the crossover point
        # Take the first half from `self`, second half from `other`
        first_half = self.genes[:crossover_point]
        second_half = other.genes[crossover_point:]
        child_genes = np.concatenate((first_half, second_half))

        return Genome(genes=child_genes, gene_set=self.gene_set)

    def mutate(self, mutation_rate: float = 0.01, *,
               mutation_amount: Optional[float] = None,
               inplace: bool = True) -> 'Genome':
        """
        Mutate the genome by altering gene values based on mutation rate and amount.

        Args:
            mutation_rate (float): Probability of each gene mutating (default is 0.01).
            mutation_amount (Optional[float]): Magnitude of mutation for real-valued genes (optional).
            inplace (bool): Whether to mutate the genome in place (default is True).

        Returns:
            Genome: The mutated genome.
        """
        if not isinstance(mutation_rate, float):
            raise TypeError(
                f'mutation_rate must be a float, found type `{type(mutation_rate)}`'
            )

        if not 0. <= mutation_rate <= 1.:
            raise ValueError(
                '`mutation_rate` must be a probability value between 0 and 1 '
                f'(both inclusive), found `{mutation_rate = }`.'
            )

        # Create a boolean mask for genes to be mutated based on mutation_rate
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
            mutated = np.clip(mutated, self.gene_range[0], self.gene_range[1])
        else:
            mutated = self._mutate_discrete(mask)

        # Apply the mutation to the genes either in-place or create a new Genome object
        if inplace:
            self.genes[mask] = mutated
            return self
        else:
            child_genes = self.genes.copy()
            child_genes[mask] = mutated
            return Genome(genes=child_genes, gene_set=self.gene_set)

    def _mutate_continuous(self, mask: np.ndarray, amount: float) -> np.ndarray:
        """
        Apply continuous mutation to the genes.
        Adds random values within [-amount, amount] to the genes selected in mask

        Args:
            mask (np.ndarray): Boolean mask indicating genes to be mutated.
            amount (float): Magnitude of mutation for real-valued genes.

        Returns:
            np.ndarray: Mutated gene values.
        """
        return self.genes[mask] + self._rng.uniform(-amount, amount, size=np.sum(mask))

    def _mutate_discrete(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply discrete mutation to the genes.
        Selects new gene values randomly from the gene set for the genes selected in mask

        Args:
            mask (np.ndarray): Boolean mask indicating genes to be mutated.

        Returns:
            np.ndarray: Mutated gene values.
        """
        return self._rng.choice(self.gene_set, size=np.sum(mask))

    def __str__(self):
        return f'Genome(genes={self.genes})'

    def __repr__(self):
        return (
            f'Genome(length={self.gene_length}, gene_set={self.gene_set}, '
            f'genes={self.genes})'
        )
