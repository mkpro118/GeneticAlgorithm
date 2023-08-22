from typing import Optional

import numpy as np

from genetic_algorithm.genome import Genome, combine_gene_set


_rng = np.random.default_rng()


def check_genome_compatibility(genome1: Genome, genome2: Genome):
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


def check_crossover_point(crossover_point: int, genome: Genome):
    if not isinstance(crossover_point, int):
        raise TypeError(
            'crossover_point must be an integer, '
            f'found type `{type(crossover_point)}`'
        )

    if not (0 < crossover_point < genome.gene_length):
        raise ValueError(
            'Crossover point is not in bounds. '
            f'There are `{genome.gene_length}` genes, '
            f'found `{crossover_point=}`'
        )


def one_point(genome1: Genome, genome2: Genome, *,
              crossover_point: Optional[int] = None) -> Genome:
    check_genome_compatibility(genome1, genome2)
    if crossover_point is not None:
        # Bounds check
        check_crossover_point(crossover_point, genome1)
    else:
        # Choose a random crossover point if one is not provided
        crossover_point = _rng.integers(1, genome1.gene_length)

    # Perform crossover by combining genes from both parents at the crossover point
    # Take the first half from `genome1`, second half from `genome2`
    first = genome1.genes[:crossover_point]
    second = genome2.genes[crossover_point:]
    child_genes = np.concatenate((first, second))

    return Genome(genes=child_genes, gene_set=combine_gene_set(genome1, genome2))


def two_point(genome1: Genome, genome2: Genome, *,
              crossover_point1: Optional[int],
              crossover_point2: Optional[int]) -> Genome:
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
    # Take the first third from `genome1`, second third from `genome2` and the last bit from `genome1` again
    first = genome1.genes[:crossover_point1]
    second = genome2.genes[crossover_point1:crossover_point2]
    third = genome1.genes[crossover_point2:]

    child_genes = np.concatenate((first, second, third))

    return Genome(genes=child_genes, gene_set=combine_gene_set(genome1, genome2))


def uniform(genome1: Genome, genome2: Genome) -> Genome:
    check_genome_compatibility(genome1, genome2)

    mask = genome1._rng.choice((False, True,), size=genome1.gene_length)
    child_genes = np.where(mask, genome1.genes, genome2.genes)

    return Genome(genes=child_genes, gene_set=genome1.gene_set)


def arithmetic(genome1: Genome, genome2: Genome, *, alpha: float = 0.5) -> Genome:
    check_genome_compatibility(genome1, genome2)

    alpha = genome1._rng.uniform(0, 1)

    child_genes = alpha * genome1.genes + (1 - alpha) * genome2.genes

    min_val = min(genome1.gene_range[0], genome2.gene_range[0])
    max_val = max(genome1.gene_range[1], genome2.gene_range[1])

    child_genes = np.clip(child_genes, min_val, max_val)

    return Genome(genes=child_genes, gene_range=(min_val, max_val))


def blend(genome1: Genome, genome2: Genome, range_factor: float) -> Genome:
    check_genome_compatibility(genome1, genome2)

    gene_ranges = np.abs(genome1 - genome2)

    min_genes = np.minimum(genome1, genome2) - gene_ranges * range_factor
    max_genes = np.maximum(genome1, genome2) + gene_ranges * range_factor

    child_genes = np.random.uniform(min_genes, max_genes, size=len(genome1))

    min_genes, max_genes = np.min(min_genes), np.max(max_genes)

    return Genome(genes=child_genes, gene_range=(min_genes, max_genes))


methods = {
    'one_point': one_point,
    'two_point': two_point,
    'uniform': uniform,
    'arithmetic': arithmetic,
    'blend': blend,
}
