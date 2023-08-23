import unittest

import string
import numpy as np

from genetic_algorithm.genome import Genome
from genetic_algorithm.crossover import (
    arithmetic,
    blend,
    one_point,
    two_point,
    uniform,
    check_genome_compatibility,
    check_crossover_point,
)


class TestCrossoverFunctions(unittest.TestCase):
    def setUp(self):
        ascii_uppercase = tuple(string.ascii_uppercase)

        self.rng = np.random.default_rng(seed=118)
        self.gene_length = int(self.rng.integers(len(ascii_uppercase)))

        self.gene_range = self.rng.uniform(1, 10), self.rng.uniform(11, 20)

        self.gene_set = self.rng.choice(ascii_uppercase,
                                        size=self.gene_length, replace=False)
        self.gene_set = tuple(self.gene_set)

        self.real_genes1 = self.rng.uniform(*self.gene_range,
                                            size=self.gene_length)
        self.real_genes2 = self.rng.uniform(*self.gene_range,
                                            size=self.gene_length)

        self.real_genome1 = Genome(genes=self.real_genes1,
                                   gene_range=self.gene_range)
        self.real_genome2 = Genome(genes=self.real_genes2,
                                   gene_range=self.gene_range)

        self.discrete_genes1 = self.rng.choice(
            self.gene_set, size=self.gene_length)
        self.discrete_genes2 = self.rng.choice(
            self.gene_set, size=self.gene_length)

        self.discrete_genome1 = Genome(genes=self.discrete_genes1,
                                       gene_set=self.gene_set)
        self.discrete_genome2 = Genome(genes=self.discrete_genes2,
                                       gene_set=self.gene_set)

        def check_blend(genes):
            return np.all(
                np.logical_and(
                    genes != self.real_genome1,
                    genes != self.real_genome2
                )
            )

        self.check_blend = check_blend
        self.join = np.concatenate

    ############################################################################
    # ##                       Arithmetic Crossover Tests                   ## #
    ############################################################################

    def test_arithmetic_with_alpha(self):
        alpha = self.rng.integers(1, 10) / 10
        alpha_ = 1 - alpha

        expected = (alpha * self.real_genes1) + (alpha_ * self.real_genes2)

        child = arithmetic(self.real_genome1, self.real_genome2, alpha=alpha)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        child = arithmetic(self.real_genome2, self.real_genome1, alpha=alpha_)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)),
                        msg=f'\n{expected=}\n{child.genes=}\n{expected!=child.genes=}')

    def test_arithmetic_with_alpha_on_self(self):
        alpha = self.rng.integers(1, 10) / 10
        alpha_ = 1 - alpha

        expected = self.real_genes1

        child = arithmetic(self.real_genome1, self.real_genome1, alpha=alpha)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)),
                        msg=f'\n{expected=}\n{child.genes=}\n{expected!=child.genes=}')

        expected = self.real_genes2

        child = arithmetic(self.real_genome2, self.real_genome2, alpha=alpha_)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

    def test_arithmetic_without_alpha(self):
        child = arithmetic(self.real_genome1, self.real_genome2)

        self.assertIsInstance(child, Genome)
        self.assertGreaterEqual(np.min(child.genes), self.gene_range[0])
        self.assertLessEqual(np.max(child.genes), self.gene_range[1])

        child = arithmetic(self.real_genome2, self.real_genome1)

        self.assertIsInstance(child, Genome)
        self.assertGreaterEqual(np.min(child.genes), self.gene_range[0])
        self.assertLessEqual(np.max(child.genes), self.gene_range[1])

    def test_arithmetic_without_alpha_on_self(self):
        child = arithmetic(self.real_genome1, self.real_genome1)

        self.assertIsInstance(child, Genome)
        self.assertGreaterEqual(np.min(child.genes), self.gene_range[0])
        self.assertLessEqual(np.max(child.genes), self.gene_range[1])

        child = arithmetic(self.real_genome2, self.real_genome2)

        self.assertIsInstance(child, Genome)
        self.assertGreaterEqual(np.min(child.genes), self.gene_range[0])
        self.assertLessEqual(np.max(child.genes), self.gene_range[1])

    ############################################################################
    # ##                        Blend Crossover Tests                       ## #
    ############################################################################

    def test_blend_with_range(self):
        range_factor = self.rng.uniform(-1, 1)

        child = blend(self.real_genome1, self.real_genome2,
                      range_factor=range_factor)

        self.assertIsInstance(child, Genome)
        self.assertTrue(self.check_blend(child.genes))

        child = blend(self.real_genome2, self.real_genome1,
                      range_factor=range_factor)

        self.assertIsInstance(child, Genome)
        self.assertTrue(self.check_blend(child.genes))

    def test_blend_with_range_on_self(self):
        range_factor = self.rng.uniform(-1, 1)

        child = blend(self.real_genome1, self.real_genome1,
                      range_factor=range_factor)

        self.assertIsInstance(child, Genome)
        self.assertTrue(self.check_blend(child.genes))

        child = blend(self.real_genome2, self.real_genome2,
                      range_factor=range_factor)

        self.assertIsInstance(child, Genome)
        self.assertTrue(self.check_blend(child.genes))

    def test_blend_without_range(self):
        child = blend(self.real_genome1, self.real_genome2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(self.check_blend(child.genes))

        child = blend(self.real_genome1, self.real_genome2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(self.check_blend(child.genes))

    def test_blend_without_range_on_self(self):
        child = blend(self.real_genome1, self.real_genome1)

        self.assertIsInstance(child, Genome)
        self.assertTrue(self.check_blend(child.genes))

        child = blend(self.real_genome2, self.real_genome2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(self.check_blend(child.genes))

    ############################################################################
    # ##                      One-Point Crossover Tests                     ## #
    ############################################################################

    def test_one_point_with_crossover_point(self):
        # ########################### Real Values ############################ #
        crossover_point = self.rng.integers(self.gene_length)

        expected = self.join((
            self.real_genes1[:crossover_point],
            self.real_genes2[crossover_point:],
        ))

        child = one_point(self.real_genome1, self.real_genome2,
                          crossover_point=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        crossover_point = self.rng.integers(self.gene_length)

        expected = self.join((
            self.real_genes2[:crossover_point],
            self.real_genes1[crossover_point:],
        ))

        child = one_point(self.real_genome2, self.real_genome1,
                          crossover_point=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        # ######################### Discrete Values ########################## #
        crossover_point = self.rng.integers(self.gene_length)

        expected = self.join((
            self.discrete_genes1[:crossover_point],
            self.discrete_genes2[crossover_point:],
        ))

        child = one_point(self.discrete_genome1, self.discrete_genome2,
                          crossover_point=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected, child.genes))

        crossover_point = self.rng.integers(self.gene_length)

        expected = self.join((
            self.discrete_genes2[:crossover_point],
            self.discrete_genes1[crossover_point:],
        ))

        child = one_point(self.discrete_genome2, self.discrete_genome1,
                          crossover_point=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected, child.genes))

    def test_one_point_with_crossover_point_on_self(self):
        # ########################### Real Values ############################ #
        crossover_point = self.rng.integers(self.gene_length)

        expected = self.real_genes1
        child = one_point(self.real_genome1, self.real_genome1,
                          crossover_point=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        crossover_point = self.rng.integers(self.gene_length)

        expected = self.real_genes2

        child = one_point(self.real_genome2, self.real_genome2,
                          crossover_point=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        # ######################### Discrete Values ########################## #
        crossover_point = self.rng.integers(self.gene_length)

        expected = self.discrete_genes1

        child = one_point(self.discrete_genome1, self.discrete_genome1,
                          crossover_point=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected, child.genes))

        crossover_point = self.rng.integers(self.gene_length)

        expected = self.discrete_genes2

        child = one_point(self.discrete_genome2, self.discrete_genome2,
                          crossover_point=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected, child.genes))

    def test_one_point_without_crossover_point(self):
        # ########################### Real Values ############################ #
        child = one_point(self.real_genome1, self.real_genome2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (
            self.real_genes1[i], self.real_genes2[i]) for i in range(len(child.genes))))

        child = one_point(self.real_genome2, self.real_genome1)

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (
            self.real_genes2[i], self.real_genes1[i]) for i in range(len(child.genes))))

        # ######################### Discrete Values ########################## #
        child = one_point(self.discrete_genome1, self.discrete_genome2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (
            self.discrete_genes1[i], self.discrete_genes2[i]) for i in range(len(child.genes))))

        child = one_point(self.discrete_genome2, self.discrete_genome1)

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (
            self.discrete_genes2[i], self.discrete_genes1[i]) for i in range(len(child.genes))))

    def test_one_point_without_crossover_point_on_self(self):
        # ########################### Real Values ############################ #
        expected = self.real_genes1
        child = one_point(self.real_genome1, self.real_genome1)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        expected = self.real_genes2

        child = one_point(self.real_genome2, self.real_genome2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        # ######################### Discrete Values ########################## #
        expected = self.discrete_genes1

        child = one_point(self.discrete_genome1, self.discrete_genome1)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected, child.genes))

        expected = self.discrete_genes2

        child = one_point(self.discrete_genome2, self.discrete_genome2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected, child.genes))

    ############################################################################
    # ##                       Two-Point Crossover Tests                    ## #
    ############################################################################

    def test_two_point_with_one_crossover_point(self):
        # ########################### Real Values ############################ #

        # ###################      Crossover Point 1      #################### #
        crossover_point = self.rng.integers(2 * self.gene_length // 3)

        expected = self.real_genes1[:crossover_point]

        child = two_point(self.real_genome1, self.real_genome2,
                          crossover_point1=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected,
                                          child.genes[:crossover_point])))
        self.assertTrue(all(child.genes[i] in (
            self.real_genes1[i], self.real_genes2[i]) for i in range(crossover_point, len(child.genes))))

        crossover_point = self.rng.integers(2 * self.gene_length // 3)

        expected = self.real_genes2[:crossover_point]

        child = two_point(self.real_genome2, self.real_genome1,
                          crossover_point1=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected,
                                          child.genes[:crossover_point])))
        self.assertTrue(all(child.genes[i] in (
            self.real_genes2[i], self.real_genes1[i]) for i in range(crossover_point, len(child.genes))))

        # ###################      Crossover Point 2      #################### #
        crossover_point = self.rng.integers(self.gene_length // 3,
                                            self.gene_length)

        expected = self.real_genes1[crossover_point:]

        child = two_point(self.real_genome1, self.real_genome2,
                          crossover_point2=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected,
                                          child.genes[crossover_point:])))
        self.assertTrue(all(child.genes[i] in (
            self.real_genes1[i], self.real_genes2[i]) for i in range(crossover_point)))

        crossover_point = self.rng.integers(self.gene_length // 3,
                                            self.gene_length)

        expected = self.real_genes2[crossover_point:]

        child = two_point(self.real_genome2, self.real_genome1,
                          crossover_point2=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected,
                                          child.genes[crossover_point:])))
        self.assertTrue(all(child.genes[i] in (
            self.real_genes2[i], self.real_genes1[i]) for i in range(crossover_point)))

        # ######################### Discrete Values ########################## #

        # ###################      Crossover Point 1      #################### #
        crossover_point = self.rng.integers(2 * self.gene_length // 3)

        expected = self.discrete_genes1[:crossover_point]

        child = two_point(self.discrete_genome1, self.discrete_genome2,
                          crossover_point1=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected,
                                       child.genes[:crossover_point]))
        self.assertTrue(all(child.genes[i] in (
            self.discrete_genes1[i], self.discrete_genes2[i]) for i in range(crossover_point, len(child.genes))))

        crossover_point = self.rng.integers(2 * self.gene_length // 3)

        expected = self.discrete_genes2[:crossover_point]

        child = two_point(self.discrete_genome2, self.discrete_genome1,
                          crossover_point1=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected,
                                       child.genes[:crossover_point]))
        self.assertTrue(all(child.genes[i] in (
            self.discrete_genes2[i], self.discrete_genes1[i]) for i in range(crossover_point, len(child.genes))))

        # ###################      Crossover Point 2      #################### #
        crossover_point = self.rng.integers(self.gene_length // 3,
                                            self.gene_length)

        expected = self.discrete_genes1[crossover_point:]

        child = two_point(self.discrete_genome1, self.discrete_genome2,
                          crossover_point2=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected,
                                       child.genes[crossover_point:]))
        self.assertTrue(all(child.genes[i] in (
            self.discrete_genes1[i], self.discrete_genes2[i]) for i in range(crossover_point)))

        crossover_point = self.rng.integers(self.gene_length // 3,
                                            self.gene_length)

        expected = self.discrete_genes2[crossover_point:]

        child = two_point(self.discrete_genome2, self.discrete_genome1,
                          crossover_point2=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected,
                                       child.genes[crossover_point:]))
        self.assertTrue(all(child.genes[i] in (
            self.discrete_genes2[i], self.discrete_genes1[i]) for i in range(crossover_point)))

    def test_two_point_with_one_crossover_point_on_self(self):
        # ########################### Real Values ############################ #

        # ###################      Crossover Point 1      #################### #
        crossover_point = self.rng.integers(2 * self.gene_length // 3)

        expected = self.real_genes1

        child = two_point(self.real_genome1, self.real_genome1,
                          crossover_point1=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        crossover_point = self.rng.integers(2 * self.gene_length // 3)

        expected = self.real_genes2

        child = two_point(self.real_genome2, self.real_genome2,
                          crossover_point1=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        # ###################      Crossover Point 2      #################### #
        crossover_point = self.rng.integers(self.gene_length // 3,
                                            self.gene_length)

        expected = self.real_genes1

        child = two_point(self.real_genome1, self.real_genome1,
                          crossover_point2=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        crossover_point = self.rng.integers(self.gene_length // 3,
                                            self.gene_length)

        expected = self.real_genes2

        child = two_point(self.real_genome2, self.real_genome2,
                          crossover_point2=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        # ######################### Discrete Values ########################## #

        # ###################      Crossover Point 1      #################### #
        crossover_point = self.rng.integers(2 * self.gene_length // 3)

        expected = self.discrete_genes1

        child = two_point(self.discrete_genome1, self.discrete_genome1,
                          crossover_point1=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected, child.genes))

        crossover_point = self.rng.integers(2 * self.gene_length // 3)

        expected = self.discrete_genes2

        child = two_point(self.discrete_genome2, self.discrete_genome2,
                          crossover_point1=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected, child.genes))

        # ###################      Crossover Point 2      #################### #
        crossover_point = self.rng.integers(self.gene_length // 3,
                                            self.gene_length)

        expected = self.discrete_genes1

        child = two_point(self.discrete_genome1, self.discrete_genome1,
                          crossover_point2=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected, child.genes))

        crossover_point = self.rng.integers(self.gene_length // 3,
                                            self.gene_length)

        expected = self.discrete_genes2

        child = two_point(self.discrete_genome2, self.discrete_genome2,
                          crossover_point2=crossover_point)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected, child.genes))

    def test_two_point_with_both_crossover_point(self):
        # ########################### Real Values ############################ #

        crossover_point1 = self.rng.integers(self.gene_length // 2)
        crossover_point2 = self.rng.integers(self.gene_length // 2,
                                             self.gene_length)

        expected = self.join((
            self.real_genes1[:crossover_point1],
            self.real_genes2[crossover_point1:crossover_point2],
            self.real_genes1[crossover_point2:],
        ))

        child = two_point(self.real_genome1, self.real_genome2,
                          crossover_point1=crossover_point1,
                          crossover_point2=crossover_point2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        crossover_point1 = self.rng.integers(self.gene_length // 2)
        crossover_point2 = self.rng.integers(self.gene_length // 2,
                                             self.gene_length)

        expected = self.join((
            self.real_genes2[:crossover_point1],
            self.real_genes1[crossover_point1:crossover_point2],
            self.real_genes2[crossover_point2:],
        ))

        child = two_point(self.real_genome2, self.real_genome1,
                          crossover_point1=crossover_point1,
                          crossover_point2=crossover_point2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        # ######################### Discrete Values ########################## #

        crossover_point1 = self.rng.integers(self.gene_length // 2)
        crossover_point2 = self.rng.integers(self.gene_length // 2,
                                             self.gene_length)

        expected = self.join((
            self.discrete_genes1[:crossover_point1],
            self.discrete_genes2[crossover_point1:crossover_point2],
            self.discrete_genes1[crossover_point2:],
        ))

        child = two_point(self.discrete_genome1, self.discrete_genome2,
                          crossover_point1=crossover_point1,
                          crossover_point2=crossover_point2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected, child.genes))

        crossover_point1 = self.rng.integers(self.gene_length // 2)
        crossover_point2 = self.rng.integers(self.gene_length // 2,
                                             self.gene_length)

        expected = self.join((
            self.discrete_genes2[:crossover_point1],
            self.discrete_genes1[crossover_point1:crossover_point2],
            self.discrete_genes2[crossover_point2:],
        ))

        child = two_point(self.discrete_genome2, self.discrete_genome1,
                          crossover_point1=crossover_point1,
                          crossover_point2=crossover_point2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected, child.genes))

    def test_two_point_with_both_crossover_point_on_self(self):
        # ########################### Real Values ############################ #

        crossover_point1 = self.rng.integers(self.gene_length // 2)
        crossover_point2 = self.rng.integers(self.gene_length // 2,
                                             self.gene_length)

        expected = self.real_genes1

        child = two_point(self.real_genome1, self.real_genome1,
                          crossover_point1=crossover_point1,
                          crossover_point2=crossover_point2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        crossover_point1 = self.rng.integers(self.gene_length // 2)
        crossover_point2 = self.rng.integers(self.gene_length // 2,
                                             self.gene_length)

        expected = self.real_genes2

        child = two_point(self.real_genome2, self.real_genome2,
                          crossover_point1=crossover_point1,
                          crossover_point2=crossover_point2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        # ######################### Discrete Values ########################## #

        crossover_point1 = self.rng.integers(self.gene_length // 2)
        crossover_point2 = self.rng.integers(self.gene_length // 2,
                                             self.gene_length)

        expected = self.discrete_genes1

        child = two_point(self.discrete_genome1, self.discrete_genome1,
                          crossover_point1=crossover_point1,
                          crossover_point2=crossover_point2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected, child.genes))

        crossover_point1 = self.rng.integers(self.gene_length // 2)
        crossover_point2 = self.rng.integers(self.gene_length // 2,
                                             self.gene_length)

        expected = self.discrete_genes2

        child = two_point(self.discrete_genome2, self.discrete_genome2,
                          crossover_point1=crossover_point1,
                          crossover_point2=crossover_point2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(expected, child.genes))

    def test_two_point_without_crossover_point(self):
        # ########################### Real Values ############################ #

        child = two_point(self.real_genome1, self.real_genome1)

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (self.real_genes1[i], self.real_genes2[i])
                            for i in range(self.gene_length)))

        child = two_point(self.real_genome2, self.real_genome2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (self.real_genes1[i], self.real_genes2[i])
                            for i in range(self.gene_length)))

        # ######################### Discrete Values ########################## #

        child = two_point(self.discrete_genome1, self.discrete_genome1)

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (self.discrete_genes1[i], self.discrete_genes2[i])
                            for i in range(self.gene_length)))

        child = two_point(self.discrete_genome2, self.discrete_genome2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (self.discrete_genes1[i], self.discrete_genes2[i])
                            for i in range(self.gene_length)))

    def test_two_point_without_crossover_point_on_self(self):
        # ########################### Real Values ############################ #

        child = two_point(self.real_genome1, self.real_genome1)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(child.genes, self.real_genes1)))

        child = two_point(self.real_genome2, self.real_genome2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(child.genes, self.real_genes2)))

        # ######################### Discrete Values ########################## #

        child = two_point(self.discrete_genome1, self.discrete_genome1)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(child.genes, self.discrete_genes1))

        child = two_point(self.discrete_genome2, self.discrete_genome2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(child.genes, self.discrete_genes2))

    ############################################################################
    # ##                        Uniform Crossover Tests                     ## #
    ############################################################################

    def test_uniform(self):
        # ########################### Real Values ############################ #

        child = uniform(self.real_genome1, self.real_genome1)

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (self.real_genes1[i], self.real_genes2[i])
                            for i in range(self.gene_length)))

        child = uniform(self.real_genome2, self.real_genome2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (self.real_genes1[i], self.real_genes2[i])
                            for i in range(self.gene_length)))

        # ######################### Discrete Values ########################## #

        child = uniform(self.discrete_genome1, self.discrete_genome1)

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (self.discrete_genes1[i], self.discrete_genes2[i])
                            for i in range(self.gene_length)))

        child = uniform(self.discrete_genome2, self.discrete_genome2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (self.discrete_genes1[i], self.discrete_genes2[i])
                            for i in range(self.gene_length)))

    def test_uniform_on_self(self):
        # ########################### Real Values ############################ #

        child = uniform(self.real_genome1, self.real_genome1)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(child.genes, self.real_genes1)))

        child = uniform(self.real_genome2, self.real_genome2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(child.genes, self.real_genes2)))

        # ######################### Discrete Values ########################## #

        child = uniform(self.discrete_genome1, self.discrete_genome1)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(child.genes, self.discrete_genes1))

        child = uniform(self.discrete_genome2, self.discrete_genome2)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.array_equal(child.genes, self.discrete_genes2))

    ############################################################################
    # ##                      Genome Compatibility Tests                    ## #
    ############################################################################

    def test_check_genome_compatibility(self):
        with self.assertRaises(ValueError):
            genome1 = Genome(gene_length=self.rng.integers(10))
            genome2 = Genome(gene_length=self.rng.integers(11, 20))

            check_genome_compatibility(genome1, genome2)

        with self.assertRaises(TypeError):
            genome1 = Genome(genes=self.rng.uniform(1, size=self.gene_length))
            genome2 = Genome(gene_length=self.gene_length, gene_set={'A', 'B'})

            check_genome_compatibility(genome1, genome2)

        with self.assertRaises(ValueError):
            genome1 = Genome(gene_set={'A', 'B'})
            genome2 = Genome(gene_set={'C', 'D'})

            check_genome_compatibility(genome1, genome2)

    def test_check_crossover_point(self):
        with self.assertRaises(TypeError):
            check_crossover_point(None, None)

        with self.assertRaises(ValueError):
            check_crossover_point(self.rng.integers(100), None)

        with self.assertRaises(ValueError):
            genome = Genome()
            check_crossover_point(genome.gene_length + 1, genome)

        with self.assertRaises(ValueError):
            genome = Genome()
            genome2 = Genome(gene_length=genome.gene_length + 1)
            check_crossover_point(genome2.gene_length + 1, genome, genome2)

    def test_crossovers_from_genome(self):
        # ############## These methods only support Real Values ############## #
        genome1 = self.real_genome1
        genome2 = self.real_genome2

        # ############################ Arithmetic ############################ #

        alpha = self.rng.integers(1, 10) / 10
        alpha_ = 1 - alpha

        expected = (alpha * genome1.genes) + (alpha_ * genome2.genes)

        child = genome1.crossover(genome2, method='arithmetic', alpha=alpha)

        self.assertIsInstance(child, Genome)
        self.assertTrue(np.all(np.isclose(expected, child.genes)))

        child = genome1.crossover(genome2, method='arithmetic')

        self.assertIsInstance(child, Genome)
        self.assertGreaterEqual(np.min(child.genes), self.gene_range[0])
        self.assertLessEqual(np.max(child.genes), self.gene_range[1])

        # ############################### Blend ############################## #

        range_factor = self.rng.uniform(-1, 1)

        child = genome1.crossover(genome2, method='blend',
                                  range_factor=range_factor)

        self.assertIsInstance(child, Genome)
        self.assertTrue(self.check_blend(child.genes))

        # ########## Methods below support Real and Discrete Values ########## #

        # Randomize
        if self.rng.random() > 0.5:
            genome1 = self.real_genome1
            genome2 = self.real_genome2
        else:
            genome1 = self.discrete_genome1
            genome2 = self.discrete_genome2

        # ############################# One-Point ############################ #
        child = genome1.crossover(genome2, method='one_point')

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (
            genome1.genes[i], genome2.genes[i]) for i in range(len(child.genes))))

        child = genome2.crossover(genome1, method='one_point')

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (
            genome2.genes[i], genome1.genes[i]) for i in range(len(child.genes))))

        # ############################# Two-Point ############################ #
        child = genome1.crossover(genome2, method='two_point')

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (
            genome1.genes[i], genome2.genes[i]) for i in range(len(child.genes))))

        child = genome2.crossover(genome1, method='two_point')

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (
            genome2.genes[i], genome1.genes[i]) for i in range(len(child.genes))))

        # ############################## Uniform ############################# #
        child = genome1.crossover(genome2, method='uniform')

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (
            genome1.genes[i], genome2.genes[i]) for i in range(len(child.genes))))

        child = genome2.crossover(genome1, method='uniform')

        self.assertIsInstance(child, Genome)
        self.assertTrue(all(child.genes[i] in (
            genome2.genes[i], genome1.genes[i]) for i in range(len(child.genes))))


if __name__ == '__main__':
    unittest.main()
