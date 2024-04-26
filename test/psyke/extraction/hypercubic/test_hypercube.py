import itertools
import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression

from psyke.extraction.hypercubic.hypercube import FeatureNotFoundException, ClosedRegressionCube, \
    ClosedClassificationCube, ClosedCube, ClassificationCube, RegressionCube, Point
from psyke.extraction.hypercubic.utils import MinUpdate, Expansion, ZippedDimension
from psyke.utils import get_int_precision
from sklearn.neighbors import KNeighborsRegressor
from psyke.extraction.hypercubic import HyperCube
from test import Predictor
from test.resources.datasets import get_dataset_path


class AbstractTestHypercube(unittest.TestCase):

    def setUp(self):
        self.x = (0.2, 0.6)
        self.y = (0.7, 0.9)
        self.dimensions = {'X': self.x, 'Y': self.y}
        self.mean = 0.5
        self.cube = HyperCube(self.dimensions, set(), self.mean)
        cubes = [({'X': (6.4, 7.9), 'Y': (5.7, 8.9)}, 5.3),
                 ({'X': (0.7, 0.8), 'Y': (0.75, 0.85)}, 6.1),
                 ({'X': (6.6, 7.0), 'Y': (9.1, 10.5)}, 7.5)]
        self.hypercubes = [HyperCube(cube[0], set(), cube[1]) for cube in cubes]
        self.dataset = pd.read_csv(get_dataset_path('arti'))
        self.filtered_dataset = self.dataset[self.dataset.apply(
            lambda row: (0.2 <= row['X'] < 0.6) & (0.7 <= row['Y'] < 0.9), axis=1)]

    def tuple_provider(self, closed=False):
        return (({'X': (self.x[0] + self.x[1]) / 2, 'Y': (self.y[0] + self.y[1]) / 2}, True),
                ({'X': self.x[0], 'Y': self.y[0]}, True),
                ({'X': self.x[0], 'Y': self.y[1]}, closed),
                ({'X': self.x[1], 'Y': self.y[0]}, closed),
                ({'X': self.x[1], 'Y': self.y[1]}, closed),
                ({'X': 1.5, 'Y': 3.6}, False))


class TestHypercube(AbstractTestHypercube):

    def test_dimension(self):
        self.assertEqual(self.dimensions, self.cube.dimensions)

    def test_limit_count(self):
        self.assertEqual(0, self.cube.limit_count)
        self.cube.add_limit('X', '+')
        self.assertEqual(1, self.cube.limit_count)
        self.cube.add_limit('Y', '-')
        self.assertEqual(2, self.cube.limit_count)
        self.cube.add_limit('X', '+')
        self.assertEqual(2, self.cube.limit_count)

    def test_output(self):
        self.assertEqual(self.mean, self.cube.output)

    def test_get(self):
        self.assertEqual(self.x, self.cube['X'])
        self.assertEqual(self.y, self.cube['Y'])
        with self.assertRaises(FeatureNotFoundException):
            _ = self.cube['Z']

    def test_get_first(self):
        self.assertEqual(self.x[0], self.cube.get_first('X'))
        self.assertEqual(self.y[0], self.cube.get_first('Y'))
        with self.assertRaises(FeatureNotFoundException):
            _ = self.cube.get_first('Z')

    def test_get_second(self):
        self.assertEqual(self.x[1], self.cube.get_second('X'))
        self.assertEqual(self.y[1], self.cube.get_second('Y'))
        with self.assertRaises(FeatureNotFoundException):
            _ = self.cube.get_second('Z')

    def test_copy(self):
        copy = self.cube.copy()
        self.assertEqual(self.cube.dimensions, copy.dimensions)
        self.assertEqual(self.cube.output, copy.output)
        self.assertIsInstance(copy, HyperCube)

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
        arguments = self.tuple_provider()
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

    def test_filter_indices(self):
        expected = (self.dataset.X >= self.x[0]) & (self.dataset.X < self.x[1]) & \
                   (self.dataset.Y >= self.y[0]) & (self.dataset.Y < self.y[1])
        filtered = self.cube.filter_indices(self.dataset.iloc[:, :-1])
        self.assertTrue(all(expected == filtered))

    def test_filter_dataframe(self):
        expected = (self.dataset.X >= self.x[0]) & (self.dataset.X < self.x[1]) & \
                   (self.dataset.Y >= self.y[0]) & (self.dataset.Y < self.y[1])
        expected = self.dataset[expected].iloc[:, :-1]
        filtered = self.cube.filter_dataframe(self.dataset.iloc[:, :-1])
        self.assertTrue(all(expected == filtered))

    def test_update(self):
        model = KNeighborsRegressor()
        model.fit(self.dataset.iloc[:, :-1], self.dataset.iloc[:, -1])
        predictor = Predictor(model)
        self.cube.update(self.dataset, predictor)
        predictions = model.predict(self.dataset[
            (self.dataset.X >= self.x[0]) & (self.dataset.X < self.x[1]) &
            (self.dataset.Y >= self.y[0]) & (self.dataset.Y < self.y[1])
        ].iloc[:, :-1])
        self.assertEqual(self.cube.output, predictions.mean())
        self.assertEqual(self.cube.diversity, predictions.std())

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

    def test_init_diversity(self):
        d = 2.3
        self.cube.init_diversity(d)
        self.assertEqual(self.cube.diversity, d)

    def test_diversity(self):
        self.assertEqual(self.cube.diversity, 0.0)
        d = 56.3
        self.cube.init_diversity(d)
        self.assertEqual(self.cube.diversity, d)

    def test_volume(self):
        self.assertEqual(self.cube.volume(), (self.x[1] - self.x[0]) * (self.y[1] - self.y[0]))

    def test_diagonal(self):
        self.assertEqual(self.cube.diagonal(), ((self.x[1] - self.x[0])**2 + (self.y[1] - self.y[0])**2)**0.5)

    def test_center(self):
        self.assertEqual(self.cube.center, Point(list(self.dimensions.keys()),
                                                 [(val[0] + val[1]) / 2 for val in self.dimensions.values()]))

    def test_corners(self):
        self.assertEqual(self.cube.corners(), [
            Point(list(self.dimensions.keys()), values) for values in itertools.product(*self.dimensions.values())])

    def test_is_adjacent(self):
        cube_adj = HyperCube({'X': (0.6, 0.9), 'Y': self.y}, set(), self.mean)
        self.assertTrue(self.cube.is_adjacent(cube_adj))
        cube_not_adj = HyperCube({'X': self.y, 'Y': self.x}, set(), self.mean)
        self.assertFalse(self.cube.is_adjacent(cube_not_adj))

    def test_merge_along_dimension(self):
        cube_adj = HyperCube({'X': (0.6, 0.9), 'Y': self.y}, set(), self.mean)
        merged = self.cube.merge_along_dimension(cube_adj, 'X')
        self.assertEqual(merged['X'][0], 0.2)
        self.assertEqual(merged['X'][1], 0.9)

    def test_zip_dimensions(self):
        cube = HyperCube({'X': self.y, 'Y': self.x})
        expected = [ZippedDimension(d, self.cube[d], cube[d]) for d in self.dimensions]
        self.assertEqual(self.cube._zip_dimensions(cube), expected)

    def test_fit_dimension(self):
        new_dimensions = {'X': (5.2, 3.6), 'Y': (9.3, 6.4)}
        self.assertEqual(self.cube._fit_dimension(new_dimensions), new_dimensions)
        new_dimensions = {'Z': (5.2, 6.4)}
        self.assertEqual(self.cube._fit_dimension(new_dimensions), new_dimensions)

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


class TestRegressionCube(AbstractTestHypercube):

    def test_copy(self):
        cube = RegressionCube(self.dimensions)
        copy = cube.copy()
        self.assertEqual(cube.dimensions, copy.dimensions)
        self.assertIsInstance(copy.output, LinearRegression)
        self.assertIsInstance(copy, RegressionCube)


class TestClassificationCube(AbstractTestHypercube):

    def test_copy(self):
        cube = ClassificationCube(self.dimensions)
        copy = cube.copy()
        self.assertEqual(cube.dimensions, copy.dimensions)
        self.assertEqual(cube.output, copy.output)
        self.assertIsInstance(copy, ClassificationCube)


class TestClosedCube(AbstractTestHypercube):

    def test_copy(self):
        cube = ClosedCube(self.dimensions)
        copy = cube.copy()
        self.assertEqual(cube.dimensions, copy.dimensions)
        self.assertEqual(cube.output, copy.output)
        self.assertIsInstance(copy, ClosedCube)

    def test_contains(self):
        cube = ClosedCube(self.dimensions)
        arguments = self.tuple_provider(True)
        for arg in arguments:
            self.assertEqual(arg[1], arg[0] in cube)


class TestClosedRegressionCube(AbstractTestHypercube):

    def test_copy(self):
        cube = ClosedRegressionCube(self.dimensions)
        copy = cube.copy()
        self.assertEqual(cube.dimensions, copy.dimensions)
        self.assertIsInstance(copy.output, LinearRegression)
        self.assertIsInstance(copy, ClosedRegressionCube)


class TestClosedClassificationCube(AbstractTestHypercube):

    def test_copy(self):
        cube = ClosedClassificationCube(self.dimensions)
        copy = cube.copy()
        self.assertEqual(cube.dimensions, copy.dimensions)
        self.assertEqual(cube.output, copy.output)
        self.assertIsInstance(copy, ClosedClassificationCube)


if __name__ == '__main__':
    unittest.main()
