from typing import Optional, Iterable, Sequence

from numpy.typing import NDArray
import numpy as np

from genetic_algorithm.utils import LazyLoader, is_int, is_real_valued_array


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

    __lazy_loader = LazyLoader()

    def __init__(self, gene_length: int = 16, *,
                 genes: Optional[Sequence | NDArray] = None,
                 gene_set: Optional[Iterable] = None,
                 gene_range: Optional[tuple[float, float]] = None,
                 random_state: Optional[int] = None):
        """
        Initialize a Genome instance.

        Parameters:
            gene_length (int): The length of the gene sequence (default is 16).
            genes (Optional[NDArray[int]]): A sequence of gene values (optional).
            gene_set (Optional[Iterable]): A set of possible gene values (optional).
            gene_range (Optional[tuple[float, float]]): The range of real-valued gene values (optional).
            random_state (Optional[int]): Seed for the random number generator (optional).
        """
        if not is_int(gene_length):
            raise TypeError(
                f'`gene_length` must be an integer, found type `{type(gene_length)=}`.'
            )
        if gene_length <= 0:
            raise ValueError(
                f'`gene_length` must be a positive number, found `{gene_length=}`.'
            )
        self._rng = np.random.default_rng(seed=random_state)

        if genes is not None:
            self.genes = np.asarray(genes)
            self.gene_length = len(genes)
        elif gene_set is not None:
            self.gene_length = gene_length
            self.genes = self._rng.choice(tuple(gene_set), size=gene_length)
        else:
            self.gene_length = gene_length
            self.genes = self._rng.integers(0, 2, size=gene_length)

        self._check_gene_set_and_range(gene_set, gene_range)

        crossover_module = Genome.__lazy_loader['genetic_algorithm.crossover']
        self._crossover_methods = crossover_module.methods

    def _check_gene_set_and_range(self, gene_set: Optional[Iterable],
                                  gene_range: Optional[tuple[float, float]]):
        """
        Validate and set the gene_set and gene_range attributes.

        Parameters:
            gene_set (Optional[Iterable]): A set of possible gene values (optional).
            gene_range (Optional[tuple[float, float]]): The range of real-valued gene values (optional).
        """
        # If the gene values are not real values
        if not is_real_valued_array(self.genes):
            # Define a gene_set for discrete values
            if gene_range is None:
                if gene_set:
                    self._gene_set = frozenset(gene_set)
                else:
                    self._gene_set = frozenset(self.genes)

                self.gene_set = tuple(self._gene_set)
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
            gene_range = (np.min(self.genes), np.max(self.genes))

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
                  method: str = 'one_point', **kwargs) -> 'Genome':
        """
        Perform crossover with another genome to create a new genome.

        Supported crossover methods are:
            - 'one_point': (Discrete and Real Values)
                In this method, a single random crossover point is selected along
                the genes of the parents. The genetic information from one parent
                from one parent is taken up to that point, and the rest is taken
                from the other parent to create the offspring.

                This method accepts one keyword argument:
                    - crossover_point (int): If provided, this value is used as
                        the crossover point, instead of a randomly generated value.


            - 'two_point': (Discrete and Real Values)
                In this method, a two random crossover points are selected along
                the genes of the parents. The genetic information from one parent
                is taken up to the first point, then genes from the other parent
                are taken upto the second point, and the rest is taken from the
                first parent again.

                This method accepts two keyword arguments:
                    - crossover_point1 (int): If provided, this value is used as
                        the first crossover point, instead of a randomly generated value.

                    - crossover_point2 (int): If provided, this value is used as
                        the second crossover point, instead of a randomly generated value.


            - 'uniform': (Discrete and Real Values)
                In this method, each bit or gene in the chromosome is chosen from
                one of the parents with a 50% probability. It's as if each gene
                is chosen with a coin toss.

                This method does not accept any keyword arguments.


            - 'arithmetic': (Real Values only)
                This method takes a weighted average of corresponding genes from
                both parents to create offspring genes.

                This method accepts one keyword argument:
                    - alpha (float): If provided, this value is used as the weight
                        fraction, instead of a randomly generated value.


            - 'blend': (Real Values only)
                This method generates offspring genes within a certain range around
                the averages of corresponding genes from both parents.

                This method accepts one keyword argument:
                    - range_factor (float): If provided, this value is used as the
                        range, instead of a randomly generated value in (-0.5, 0.5).

        Parameters:
            other (Genome): The other genome to perform crossover with.
            method (str): The crossover method to use. Default is 'one_point'.
            kwargs: optional keyword arguments for crossover methods

        Returns:
            Genome: A new genome resulting from crossover.
        """
        try:
            return self._crossover_methods[method](self, other, **kwargs)
        except KeyError:
            raise ValueError(
                f'`{method=}` is not a known crossover method'
            )

    def mutate(self, mutation_rate: float = 0.01, *,
               mutation_amount: Optional[float] = None,
               inplace: bool = True) -> 'Genome':
        """
        Mutate the genome by altering gene values based on mutation rate and amount.

        Parameters:
            mutation_rate (float): Probability of each gene mutating (default is 0.01).
            mutation_amount (Optional[float]): Magnitude of mutation for real-valued genes (optional).
            inplace (bool): Whether to mutate the genome in place (default is True).

        Returns:
            Genome: The mutated genome.
        """
        if not isinstance(mutation_rate, float):
            raise TypeError(
                f'mutation_rate must be a float between 0 and 1, found type `{type(mutation_rate)}`'
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

        Parameters:
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

        Parameters:
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


def combine_gene_set(genome1: Genome, genome2: Genome) -> frozenset:
    """
    Combine gene sets from two genomes.

    This function takes two Genome objects and returns a frozenset containing
    the combined gene set of both genomes. Gene sets are returned as
    frozensets to ensure immutability.

    Parameters:
        genome1 (Genome): The first Genome object containing gene information.
        genome2 (Genome): The second Genome object containing gene information.

    Returns:
        frozenset: A frozenset containing the combined gene set from both genomes.
    """
    return genome1._gene_set.union(genome2._gene_set)


def combine_gene_range(genome1: Genome, genome2: Genome) -> tuple[float, float]:
    """
    Combine the gene ranges of two genomes.

    This function computes the combined gene range by taking the minimum of the
    lower bounds and the maximum of the upper bounds from the gene ranges of
    the given genomes.

    Parameters:
        genome1 (Genome): The first genome.
        genome2 (Genome): The second genome.

    Returns:
        tuple: A tuple containing the combined gene range, where the first element
               is the lower bound and the second element is the upper bound.
    """
    return (
        min(genome1.gene_range[0], genome2.gene_range[0]),
        max(genome1.gene_range[1], genome2.gene_range[1]),
    )
