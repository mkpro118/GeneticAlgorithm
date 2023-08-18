from typing import Optional

import numpy as np

from genetic_algorithm.genome import Genome


def check_genome_compatibility(genome1, genome2):
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


def one_point(genome1: Genome, genome2: Genome, *,
              crossover_point: Optional[int] = None) -> Genome:
    if crossover_point is not None:
        if not isinstance(crossover_point, int):
            raise TypeError(
                'crossover_point must be an integer, '
                f'found type `{type(crossover_point)}`'
            )
        if not (0 < crossover_point < genome1.gene_length):
            raise ValueError(
                'Crossover point is not in bounds. '
                f'There are `{genome1.gene_length}` genes, but '
                f'crossover point was found to be `{crossover_point}`'
            )
    else:
        # Choose a random crossover point if one is not provided
        crossover_point = genome1._rng.integers(1, genome1.gene_length)

    # Perform crossover by combining genes from both parents at the crossover point
    # Take the first half from `genome1`, second half from `genome2`
    first_half = genome1.genes[:crossover_point]
    second_half = genome2.genes[crossover_point:]
    child_genes = np.concatenate((first_half, second_half))

    return Genome(genes=child_genes, gene_set=genome1.gene_set)


def two_point(genome1: Genome, genome2: Genome, *,
              crossover_point1: Optional[int],
              crossover_point2: Optional[int]) -> Genome:
    check_genome_compatibility(genome1, genome2)
    pass


def uniform(genome1: Genome, genome2: Genome) -> Genome:
    check_genome_compatibility(genome1, genome2)
    pass


def arithmetic(genome1: Genome, genome2: Genome, alpha: float) -> Genome:
    check_genome_compatibility(genome1, genome2)
    pass


def blend(genome1: Genome, genome2: Genome, range_factor: float) -> Genome:
    check_genome_compatibility(genome1, genome2)
    pass


def discrete(genome1: Genome, genome2: Genome) -> Genome:
    check_genome_compatibility(genome1, genome2)
    pass


methods = {
    'one_point': one_point,
    'two_point': two_point,
    'uniform': uniform,
    'arithmetic': arithmetic,
    'blend': blend,
    'discrete': discrete,
}
