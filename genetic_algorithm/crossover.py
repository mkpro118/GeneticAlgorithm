from typing import Optional

import numpy as np

from genetic_algorithm.genome import Genome, combine_gene_set
from genetic_algorithm.utils import kwargs_proxy


__all__ = (
    'methods',
    'arithmetic',
    'blend',
    'one_point',
    'two_point',
    'uniform',
    'check_genome_compatibility',
    'check_crossover_point',
)

_rng = np.random.default_rng()


def check_genome_compatibility(genome1: Genome, genome2: Genome):
    """
    Check the compatibility of two genomes for crossover.

    This function checks whether two given genomes are compatible for crossover
    by comparing their gene lengths and gene sets. If the gene lengths or gene sets
    do not match, a ValueError is raised.

    Parameters:
        genome1 (Genome): The first genome to be checked for compatibility.
        genome2 (Genome): The second genome to be checked for compatibility.

    Raises:
        ValueError: If the gene lengths of the genomes are not the same,
                    or if the gene sets of the genomes do not match.
    """
    if genome1.gene_length != genome2.gene_length:
        raise ValueError(
            f'Genes are not of the same length, '
            f'{genome1.gene_length} != {genome2.gene_length}.'
        )

    if genome1._gene_set != genome2._gene_set:
        raise ValueError(
            f'Gene sets of the genomes do not match. '
            f'Gene set of genome1: {genome1._gene_set}. '
            f'Gene set of genome2: {genome2._gene_set}.'
        )


def check_crossover_point(crossover_point: int, genome: Genome,
                          genome2: Optional[Genome] = None):
    """
    Check the validity of a crossover point for a given genome.

    This function checks whether the provided crossover point is valid for the given genome
    and an optional second genome. The crossover point must be within bounds and an integer.

    Note:
        This function does NOT check whether the given genomes are of equal length.
        A crossover will be considered valid on genomes of unequal length, if the
        crossover point is a value lesser than the length of the smaller genome.

    Parameters:
        crossover_point (int): The proposed crossover point for the genomes.
        genome (Genome): The genome for which the crossover point is being checked.
        genome2 (Optional[Genome]): An optional second genome for comparison.
                                   If not provided, the function will use the same genome.

    Raises:
        TypeError: If the crossover point is not an integer.
        ValueError: If the genome is None, or if the crossover point is out of bounds.
    """
    if not isinstance(crossover_point, int):
        raise TypeError(
            'crossover_point must be an integer, '
            f'found type `{type(crossover_point)}`'
        )

    if genome is None:
        raise ValueError(f'Cannot check crossover point as genome is {None}.')

    # Genomes are used for length checks. If a second genome2 is not provided,
    # we simply use the first genome twice.
    genome2 = genome2 or genome

    # For a valid crossover, the crossover point must be lesser than the length
    # of the smaller genome.
    length = min(genome.gene_length, genome2.gene_length)

    if not (0 < crossover_point < length):
        raise ValueError(
            'Crossover point is not in bounds. '
            f'There are `{length}` genes, '
            f'found `{crossover_point=}`.'
        )


@kwargs_proxy
def arithmetic(genome1: Genome, genome2: Genome, *,
               alpha: Optional[float] = None) -> Genome:
    """
    Perform arithmetic crossover between two genomes.

    This function performs arithmetic crossover by creating a child genome where each gene
    is a weighted average of the corresponding genes from the two parent genomes. The weight
    fraction is determined by the alpha value, which is a user-defined parameter between 0 and 1.
    The child genome's genes are then clipped to fall within the valid gene range defined by
    the parent genomes.

    Parameters:
        genome1 (Genome): The first parent genome.
        genome2 (Genome): The second parent genome.
        alpha (Optional[float]): The alpha value for weighted averaging of genes.
                                If not provided, a random value between 0 and 1 is used.

    Returns:
        Genome: The child genome resulting from arithmetic crossover.

    Raises:
        ValueError: If the gene lengths of the genomes are not compatible.
    """
    check_genome_compatibility(genome1, genome2)

    # Generate a random alpha value between 0 and 1.
    # Alpha is the weight fraction for a weighted average.
    alpha = alpha or _rng.uniform(0, 1)

    # Combine genes using the arithmetic crossover formula with the calculated alpha value.
    child_genes = alpha * genome1.genes + (1 - alpha) * genome2.genes

    # Determine the minimum and maximum possible gene values from the parent genomes.
    # This forms the gene range for the child's genome
    min_val = min(genome1.gene_range[0], genome2.gene_range[0])
    max_val = max(genome1.gene_range[1], genome2.gene_range[1])

    # Clip the child genes to ensure they fall within the valid gene range.
    child_genes = np.clip(child_genes, min_val, max_val)

    return Genome(genes=child_genes, gene_range=(min_val, max_val))


@kwargs_proxy
def blend(genome1: Genome, genome2: Genome, *,
          range_factor: Optional[float] = None) -> Genome:
    """
    Perform blend crossover between two genomes.

    This function performs blend crossover by generating a child genome with genes that are
    uniformly distributed within a range calculated from the parent genomes. The range is
    expanded by a factor specified by the range_factor parameter. The child genome's gene
    values are clipped to fall within the valid gene range defined by the parent genomes.

    Parameters:
        genome1 (Genome): The first parent genome.
        genome2 (Genome): The second parent genome.
        range_factor (float): The factor by which the gene range is expanded.

    Returns:
        Genome: The child genome resulting from blend crossover.

    Raises:
        ValueError: If the gene lengths of the genomes are not compatible.
    """
    check_genome_compatibility(genome1, genome2)

    range_factor = range_factor or _rng.uniform(-0.5, 0.5)

    # Calculate the absolute gene ranges for each gene in genome1 and genome2.
    gene_ranges = np.abs(genome1 - genome2)

    # Calculate the minimum and maximum possible gene values for the child genome,
    # considering the range factor.
    min_genes = np.minimum(genome1, genome2) - gene_ranges * range_factor
    max_genes = np.maximum(genome1, genome2) + gene_ranges * range_factor

    # Generate random child genes using uniform distribution within the calculated ranges.
    child_genes = np.random.uniform(min_genes, max_genes, size=len(genome1))

    # Determine the range of gene values for the child genome.
    min_genes, max_genes = np.min(child_genes), np.max(child_genes)

    return Genome(genes=child_genes, gene_range=(min_genes, max_genes))


@kwargs_proxy
def one_point(genome1: Genome, genome2: Genome, *,
              crossover_point: Optional[int] = None) -> Genome:
    """
    Perform one-point crossover between two genomes.

    This function combines genes from two parent genomes at a specified crossover point
    or a randomly chosen point if not provided. The resulting child genome inherits the
    first part of genes from the first parent and the second part from the second parent.

    Parameters:
        genome1 (Genome): The first parent genome.
        genome2 (Genome): The second parent genome.
        crossover_point (Optional[int]): The crossover point where genes are exchanged.
                                         If not provided, a random point is chosen.

    Returns:
        Genome: The child genome resulting from the crossover.

    Raises:
        TypeError: If the crossover point is not an integer.
        ValueError: If the gene lengths of the genomes are not compatible.
        ValueError: If the crossover point is out of bounds.
    """
    check_genome_compatibility(genome1, genome2)

    if crossover_point is not None:
        # Bounds check
        check_crossover_point(crossover_point, genome1)
    else:
        # Choose a random crossover point if one is not provided
        crossover_point = _rng.integers(1, genome1.gene_length)

    # Perform crossover by combining genes from both parents at the crossover point
    # Take the first half from `genome1`
    first = genome1.genes[:crossover_point]

    # Take second half from `genome2`
    second = genome2.genes[crossover_point:]

    # Combine into a single genome
    child_genes = np.concatenate((first, second))

    return Genome(genes=child_genes, gene_set=combine_gene_set(genome1, genome2))


@kwargs_proxy
def two_point(genome1: Genome, genome2: Genome, *,
              crossover_point1: Optional[int],
              crossover_point2: Optional[int]) -> Genome:
    """
    Perform two-point crossover between two genomes.

    This function combines genes from two parent genomes at two specified crossover points
    or randomly chosen points if not provided. The resulting child genome inherits genes
    from the first parent up to the first crossover point, genes from the second parent
    between the two crossover points, and genes from the first parent after the second
    crossover point.

    Parameters:
        genome1 (Genome): The first parent genome.
        genome2 (Genome): The second parent genome.
        crossover_point1 (Optional[int]): The first crossover point where genes are exchanged.
        crossover_point2 (Optional[int]): The second crossover point where genes are exchanged.

    Returns:
        Genome: The child genome resulting from the two-point crossover.

    Raises:
        TypeError: If either crossover point is not an integer.
        ValueError: If the gene lengths of the genomes are not compatible.
        ValueError: If either crossover point is out of bounds.
    """
    check_genome_compatibility(genome1, genome2)

    if crossover_point1 is not None:
        # Bounds check
        check_crossover_point(crossover_point1, genome1)
    else:
        # Choose a random crossover point if one is not provided
        crossover_point1 = _rng.integers(1, 2 * genome1.gene_length // 3)

    if crossover_point2 is not None:
        # Bounds check
        check_crossover_point(crossover_point2, genome2)
    else:
        # Choose a random crossover point if one is not provided
        crossover_point2 = _rng.integers(crossover_point1 + 1,
                                         genome2.gene_length)

    # Perform crossover by combining genes from both parents at the two crossover points
    # Take genes upto the first crosover point from `genome1`, then,
    first = genome1.genes[:crossover_point1]

    # Take genes upto the second crossover point from `genome2` then,
    second = genome2.genes[crossover_point1:crossover_point2]

    # Take the remaining genes from `genome1` again
    third = genome1.genes[crossover_point2:]

    # Combine into a single genome
    child_genes = np.concatenate((first, second, third))

    return Genome(genes=child_genes, gene_set=combine_gene_set(genome1, genome2))


@kwargs_proxy
def uniform(genome1: Genome, genome2: Genome) -> Genome:
    """
    Perform uniform crossover between two genomes.

    This function performs uniform crossover by creating a random mask of boolean values,
    where True corresponds to genes from the first parent genome (genome1), and False
    corresponds to genes from the second parent genome (genome2). The child genome is
    then created by selecting genes from either genome1 or genome2 based on the mask.

    Parameters:
        genome1 (Genome): The first parent genome.
        genome2 (Genome): The second parent genome.

    Returns:
        Genome: The child genome resulting from uniform crossover.

    Raises:
        ValueError: If the gene lengths of the genomes are not compatible.
    """
    check_genome_compatibility(genome1, genome2)

    # Generate a random boolean mask where,
    # True corresponds to genes from genome1
    # False corresponds to genes from genome2.
    mask = genome1._rng.choice((False, True,), size=genome1.gene_length)

    # Combine genes based on the mask, selecting genes from either genome1 or genome2.
    child_genes = np.where(mask, genome1.genes, genome2.genes)

    return Genome(genes=child_genes, gene_set=genome1.gene_set)


# Dictionary of crossover methods and their corresponding functions
methods = {
    'arithmetic': arithmetic,
    'blend': blend,
    'one_point': one_point,
    'two_point': two_point,
    'uniform': uniform,
}
