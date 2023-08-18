import unittest
import numpy as np
from genetic_algorithm.genome import Genome


class TestGenome(unittest.TestCase):
    def test_init_default(self):
        genome = Genome()
        self.assertEqual(len(genome.genes), 16)
        self.assertEqual(genome.gene_length, 16)
        self.assertTrue(hasattr(genome, 'gene_set'))
        self.assertFalse(hasattr(genome, 'gene_range'))

    def test_init_invalid_gene_length(self):
        with self.assertRaises(ValueError):
            Genome(gene_length=0)

        with self.assertRaises(ValueError):
            Genome(gene_length=-1)

    def test_init_invalid_gene_length_type(self):
        with self.assertRaises(TypeError):
            Genome(gene_length=print)

    def test_init_with_genes(self):
        genes = [0, 1, 0, 1, 1, 0, 0, 1]
        genome = Genome(genes=genes)
        self.assertTrue(np.array_equal(genome.genes, np.array(genes)))
        self.assertEqual(genome.gene_length, len(genes))

    def test_init_with_gene_set(self):
        gene_length = 5
        gene_set = frozenset({'A', 'B'})
        genome = Genome(gene_length=gene_length, gene_set=gene_set)

        self.assertEqual(genome.gene_length, gene_length)
        self.assertEqual(genome._gene_set, gene_set)
        self.assertTrue(all(gene in gene_set for gene in genome.genes))

    def test_init_continuous(self):
        genes = np.array([0.1, 0.5, 0.8])
        gene_range = (0.0, 1.0)
        genome = Genome(genes=genes, gene_range=gene_range)
        self.assertTrue(np.array_equal(genome.genes, genes))
        self.assertEqual(genome.gene_range, gene_range)

    def test_init_discrete_with_range(self):
        with self.assertRaises(ValueError):
            Genome(genes={'10', '10', '10'}, gene_range=(5, 10))

    def test_init_continuous_without_range(self):
        with self.assertRaises(ValueError):
            Genome(genes=np.linspace(0, 1))

    def test_init_continuous_with_invalid_range_length(self):
        with self.assertRaises(ValueError):
            Genome(genes=np.linspace(0, 1), gene_range=(1, 2, 3))

    def test_init_continuous_with_invalid_range_type(self):
        with self.assertRaises(ValueError):
            Genome(genes=np.linspace(0, 1), gene_range=(map, filter))

    def test_crossover_normal(self):
        parent1 = Genome(genes=[0, 1, 0, 1, 0, 1, 0, 1])
        parent2 = Genome(genes=[1, 0, 1, 0, 1, 0, 1, 0])
        crossover_point = 4
        child = parent1.crossover(parent2, crossover_point=crossover_point)
        expected_genes = [0, 1, 0, 1, 1, 0, 1, 0]
        self.assertTrue(np.array_equal(child.genes, np.array(expected_genes)))

    def test_crossover_length_mismatch(self):
        parent1 = Genome(genes=[0, 1, 0, 1, 0, 1, 0, 1])
        parent2 = Genome(genes=[1, 0, 1, 0, 1, 0, 1])
        with self.assertRaises(ValueError):
            parent1.crossover(parent2)

    def test_crossover_gene_set_mismatch(self):
        parent1 = Genome(genes=[0, 1, 0, 1, 0, 1, 0, 1])
        parent2 = Genome(genes=[1, 2, 5, 1, 0, 1, 0, 1])
        with self.assertRaises(ValueError):
            parent1.crossover(parent2)

    def test_crossover_invalid_crossover_point_type(self):
        parent1 = Genome(genes=[0, 1, 0, 1, 0, 1, 0, 1])
        parent2 = Genome(genes=[1, 0, 1, 0, 1, 0, 1, 0])
        with self.assertRaises(TypeError):
            parent1.crossover(parent2, crossover_point='1')

    def test_crossover_invalid_crossover_point_value(self):
        parent1 = Genome(genes=[0, 1, 0, 1, 0, 1, 0, 1])
        parent2 = Genome(genes=[1, 0, 1, 0, 1, 0, 1, 0])
        with self.assertRaises(ValueError):
            parent1.crossover(parent2, crossover_point=10)

    def test_mutate_invalid_mutation_rate_type(self):
        genes = [0, 1, 0, 1, 0, 1, 0, 1]
        genome = Genome(genes=genes)
        with self.assertRaises(TypeError):
            genome.mutate(mutation_rate='1.0')

    def test_mutate_invalid_mutation_rate_value(self):
        genes = [0, 1, 0, 1, 0, 1, 0, 1]
        genome = Genome(genes=genes)
        with self.assertRaises(ValueError):
            genome.mutate(mutation_rate=1.1)

    def test_mutate_invalid_mutation_amount_type(self):
        genes = [0., 1., 0., 1.]
        genome = Genome(genes=genes, gene_range=(0, 1))
        with self.assertRaises(TypeError):
            genome.mutate(mutation_rate=0.5, mutation_amount='1.0')

    def test_mutate_invalid_mutation_amount_value(self):
        genes = [0., 1., 0., 1.]
        genome = Genome(genes=genes, gene_range=(0, 1))
        with self.assertRaises(ValueError):
            genome.mutate(mutation_rate=0.5)

    def test_mutate_discrete(self):
        genes = [0, 1, 0, 1, 0, 1, 0, 1]
        genome = Genome(genes=genes)
        mutated_genome = genome.mutate(mutation_rate=0.0)
        self.assertTrue(np.array_equal(mutated_genome.genes, genes))

    def test_mutate_continuous(self):
        genes = np.array([0.2, 0.5, 0.8])
        gene_range = (0.0, 1.0)
        genome = Genome(genes=genes, gene_range=gene_range)
        mutated_genome = genome.mutate(mutation_rate=0.0, mutation_amount=0.1)
        self.assertTrue(np.array_equal(mutated_genome.genes, genes))

    def test_str(self):
        genes = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        genome = Genome(genes=genes)
        self.assertEqual(str(genome), f'Genome(genes={genes})')

    def test_repr(self):
        genes = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        genome = Genome(genes=genes)
        self.assertEqual(
            repr(genome), f'Genome(length={len(genes)}, gene_set={tuple(set(genes.tolist()))}, genes={genes})')


if __name__ == '__main__':
    unittest.main()
