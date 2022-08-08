import unittest
import pandas as pd

from psyke.extraction.hypercubic.hypercube import FeatureNotFoundException
from psyke.extraction.hypercubic.utils import MinUpdate, Expansion
from psyke.utils import get_int_precision
from sklearn.neighbors import KNeighborsRegressor
from test.psyke import Predictor
from psyke.extraction.hypercubic import HyperCube
from test.resources.datasets import get_dataset_path


class AbstractTestHypercube(unittest.TestCase):

    def setUp(self):
        self.dimensions = {'X': (0.2, 0.6), 'Y': (0.7, 0.9)}
        self.mean = 0.5
        self.cube = HyperCube(self.dimensions, set(), self.mean)
        cubes = [({'X': (6.4, 7.9), 'Y': (5.7, 8.9)}, 5.3),
                 ({'X': (0.7, 0.8), 'Y': (0.75, 0.85)}, 6.1),
                 ({'X': (6.6, 7.0), 'Y': (9.1, 10.5)}, 7.5)]
        self.hypercubes = [HyperCube(cube[0], set(), cube[1]) for cube in cubes]
        self.dataset = pd.read_csv(get_dataset_path('arti'))
        self.filtered_dataset = self.dataset[self.dataset.apply(
            lambda row: (0.2 <= row['X'] < 0.6) & (0.7 <= row['Y'] < 0.9), axis=1)]


class TestHypercube(AbstractTestHypercube):

    def test_get_dimension(self):
        self.assertEqual(self.dimensions, self.cube.dimensions)

    def test_get_limit_count(self):
        self.assertEqual(0, self.cube.limit_count)
        self.cube.add_limit('X', '+')
        self.assertEqual(1, self.cube.limit_count)
        self.cube.add_limit('Y', '-')
        self.assertEqual(2, self.cube.limit_count)
        self.cube.add_limit('X', '+')
        self.assertEqual(2, self.cube.limit_count)

    def test_get_mean(self):
        self.assertEqual(self.mean, self.cube.output)

    def test_get(self):
        self.assertEqual((0.2, 0.6), self.cube['X'])
        self.assertEqual((0.7, 0.9), self.cube['Y'])
        with self.assertRaises(FeatureNotFoundException):
            dummy = self.cube['Z']

    def test_get_first(self):
        self.assertEqual(0.2, self.cube.get_first('X'))
        self.assertEqual(0.7, self.cube.get_first('Y'))
        with self.assertRaises(FeatureNotFoundException):
            self.cube.get_first('Z')

    def test_get_second(self):
        self.assertEqual(0.6, self.cube.get_second('X'))
        self.assertEqual(0.9, self.cube.get_second('Y'))
        with self.assertRaises(FeatureNotFoundException):
            self.cube.get_second('Z')

    def test_copy(self):
        copy = self.cube.copy()
        self.assertEqual(self.cube.dimensions, copy.dimensions)
        self.assertEqual(self.cube.output, copy.output)

    def test_expand(self):
        arguments = TestHypercube.expansion_provider()
        for arg in arguments:
            arg[0].expand(arg[1], self.hypercubes)
            self.assertEqual(arg[2], arg[0][arg[1].feature])

    def test_expand_all(self):
        updates = [MinUpdate('X', 0.2), MinUpdate('Y', 0.15)]
        surrounding = HyperCube({'X': (0.0, 0.8), 'Y': (0.1, 0.6)}, output=0.0)
        cube = HyperCube({'X': (0.1, 0.2), 'Y': (0.4, 0.4)}, output=0.4)
        cube.expand_all(updates, surrounding)
        self.assertEqual((0.0, 0.4), cube.dimensions['X'])
        self.assertEqual((0.25, 0.55), cube.dimensions['Y'])

    def test_overlap(self):
        self.assertIsNone(self.cube.overlap(self.hypercubes))
        self.assertFalse(self.cube.overlap(self.hypercubes[0]))
        self.assertFalse(self.cube.overlap(self.hypercubes[1]))
        self.cube.update_dimension('X', 0.6, 1.0)
        self.assertIsNotNone(self.cube.overlap(self.hypercubes))
        self.assertEqual(self.hypercubes[1], self.cube.overlap(self.hypercubes))
        self.assertFalse(self.cube.overlap(self.hypercubes[0]))
        self.assertTrue(self.cube.overlap(self.hypercubes[1]))
        self.assertTrue(self.hypercubes[1].overlap(self.cube))

    def test_has_volume(self):
        self.assertTrue(self.cube.has_volume())
        no_volume = self.cube.copy()
        no_volume.update_dimension('X', 1.0, 1.0)
        self.assertFalse(no_volume.has_volume())

    def test_equal(self):
        self.assertTrue(self.cube == self.cube)
        self.assertFalse(self.cube == self.hypercubes[1])

    def test_contains(self):
        arguments = TestHypercube.tuple_provider()
        for arg in arguments:
            self.assertEqual(arg[1], arg[0] in self.cube)

    def test_count(self):
        self.assertEqual(self.dataset.shape[0], HyperCube.create_surrounding_cube(self.dataset).count(self.dataset))
        self.assertEqual(self.filtered_dataset.shape[0], self.cube.count(self.dataset))

    def test_create_samples(self):
        points = self.cube.create_samples(25)
        for k, v in self.cube.dimensions.items():
            self.assertTrue(all((points.loc[:, k] >= v[0]).values))
            self.assertTrue(all((points.loc[:, k] < v[1]).values))

    def test_add_limit(self):
        self.assertEqual(0, self.cube.limit_count)
        self.cube.add_limit('X', '-')
        self.assertEqual(1, self.cube.limit_count)
        self.cube.add_limit('X', '-')
        self.assertEqual(1, self.cube.limit_count)
        self.cube.add_limit('X', '+')
        self.assertEqual(2, self.cube.limit_count)
        self.cube.add_limit('X', '+')
        self.assertEqual(2, self.cube.limit_count)

    def test_check_limits(self):
        self.assertIsNone(self.cube.check_limits('X'))
        self.cube.add_limit('X', '-')
        self.assertEqual('-', self.cube.check_limits('X'))
        self.cube.add_limit('X', '+')
        self.assertEqual('*', self.cube.check_limits('X'))
        self.assertIsNone(self.cube.check_limits('Y'))
        self.cube.add_limit('Y', '+')
        self.assertEqual('+', self.cube.check_limits('Y'))
        self.cube.add_limit('Y', '-')
        self.assertEqual('*', self.cube.check_limits('Y'))

    def test_update_mean(self):
        model = KNeighborsRegressor()
        model.fit(self.dataset.iloc[:, :-1], self.dataset.iloc[:, -1])
        predictor = Predictor(model)
        self.cube.update(self.dataset, predictor)

    def test_update_dimension(self):
        new_lower, new_upper = 0.6, 1.4
        updated = {'X': (new_lower, new_upper),
                   'Y': (0.7, 0.9)}
        new_cube1 = self.cube.copy()
        new_cube1.update_dimension('X', new_lower, new_upper)
        self.assertEqual(updated, new_cube1.dimensions)
        new_cube2 = self.cube.copy()
        new_cube2.update_dimension('X', (new_lower, new_upper))
        self.assertEqual(updated, new_cube2.dimensions)

    def test_create_surrounding_cube(self):
        surrounding = HyperCube.create_surrounding_cube(self.dataset)
        for feature in self.dataset.columns[:-1]:
            self.assertEqual((round(min(self.dataset[feature]) - HyperCube.EPSILON * 2, get_int_precision()),
                              round(max(self.dataset[feature]) + HyperCube.EPSILON * 2, get_int_precision())),
                             surrounding.dimensions[feature])

    def test_cube_from_point(self):
        lower, upper, mean = 0.5, 0.8, 0.6
        cube = HyperCube.cube_from_point({'X': lower, 'Y': upper, 'z': mean})
        self.assertEqual({'X': (lower, lower), 'Y': (upper, upper)}, cube.dimensions)
        self.assertEqual(mean, cube.output)

    def test_check_overlap(self):
        self.assertTrue(HyperCube.check_overlap((self.hypercubes[0], ), (self.hypercubes[0].copy(), )))
        self.assertTrue(HyperCube.check_overlap(self.hypercubes, self.hypercubes + [self.hypercubes[0].copy()]))
        self.assertFalse(HyperCube.check_overlap(self.hypercubes, self.hypercubes))
        self.assertFalse(HyperCube.check_overlap(self.hypercubes[0:1], self.hypercubes[1:]))

    @staticmethod
    def expansion_provider():
        cube1 = HyperCube({'X': (2.3, 6.4), 'Y': (8.9, 12.3)}, output=2.3)
        fake1 = cube1.copy()
        fake1.update_dimension('X', 0.5, 2.3)
        fake2 = cube1.copy()
        fake2.update_dimension('X', 6.4, 12.9)
        cube2 = cube1.copy()
        cube2.update_dimension('X', 9.5, 12.3)
        fake3 = cube2.copy()
        fake3.update_dimension('X', 5.0, 9.5)
        fake4 = cube2.copy()
        fake4.update_dimension('X', 12.3, 15.2)

        return [(cube1.copy(), Expansion(fake1, 'X', '-', 0.0), (0.5, 6.4)),
                (cube1.copy(), Expansion(fake2, 'X', '+', 0.0), (2.3, 6.6)),
                (cube2.copy(), Expansion(fake3, 'X', '-', 0.0), (7.0, 12.3)),
                (cube2.copy(), Expansion(fake4, 'X', '+', 0.0), (9.5, 15.2))]

    @staticmethod
    def tuple_provider():
        return (({'X': 0.5, 'Y': 0.8}, True),
                ({'X': 0.1, 'Y': 0.8}, False),
                ({'X': 0.5, 'Y': 0.95}, False),
                ({'X': 0.1, 'Y': 0.95}, False))


if __name__ == '__main__':
    unittest.main()
