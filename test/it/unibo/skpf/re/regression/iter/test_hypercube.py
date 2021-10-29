import unittest

import pandas as pd

from psyke.regression.feature_not_found_exception import FeatureNotFoundException
from psyke.regression.hypercube import HyperCube
from psyke.regression.iter.minupdate import MinUpdate


class TestUtility:

    @staticmethod
    def create_cube(x: tuple, y: tuple, output: float) -> HyperCube:
        return HyperCube({'X': x, 'Y': y}, output=output)

    @staticmethod
    def get_cubes() -> list[HyperCube]:
        hypercubes = [((6.4, 7.9), (5.7, 8.9), 5.3),
                      ((0.7, 0.8), (0.75, 0.85), 6.1),
                      ((6.6, 7.0), (9.1, 10.5), 7.5)]
        return [TestUtility.create_cube(it[0], it[1], it[2]) for it in hypercubes]

    @staticmethod
    def get_cube() -> HyperCube:
        dimensions = TestUtility.get_dimensions()
        return TestUtility.create_cube(dimensions['X'], dimensions['Y'], TestUtility.get_mean())

    @staticmethod
    def get_dimensions() -> dict[str, tuple]:
        return {'X': (0.2, 0.6), 'Y': (0.7, 0.9)}

    @staticmethod
    def get_mean() -> float:
        return 0.5


class TestHypercube(unittest.TestCase):
    cube = TestUtility.get_cube()
    hypercubes = TestUtility.get_cubes()
    dimensions = TestUtility.get_dimensions()
    mean = TestUtility.get_mean()
    dataset = pd.read_csv('resources/arti.csv')
    filtered_dataset = dataset[dataset.apply(lambda row: (0.2 <= row['X'] < 0.6) & (0.7 <= row['Y'] < 0.9), axis=1)]

    def test_get_dimension(self):
        self.assertEqual(self.dimensions, self.cube.dimensions)

    def test_get_limit_count(self):
        self.assertEqual(0, self.cube.limit_count)
        self.cube.add_limit("X", '+')
        self.assertEqual(1, self.cube.limit_count)
        self.cube.add_limit("Y", '-')
        self.assertEqual(2, self.cube.limit_count)
        self.cube.add_limit("X", '+')
        self.assertEqual(2, self.cube.limit_count)

    def test_get_mean(self):
        self.assertEqual(self.mean, self.cube.mean)

    def test_get(self):
        self.assertEqual((0.2, 0.6), self.cube.get("X"))
        self.assertEqual((0.7, 0.9), self.cube.get("Y"))
        self.assertRaises(FeatureNotFoundException, self.cube.get("Z"))

    def test_get_first(self):
        self.assertEqual(0.2, self.cube.get_first("X"))
        self.assertEqual(0.7, self.cube.get_first("Y"))
        self.assertRaises(FeatureNotFoundException, self.cube.get_first("Z"))

    def test_get_second(self):
        self.assertEqual(0.6, self.cube.get_second("X"))
        self.assertEqual(0.9, self.cube.get_second("Y"))
        self.assertRaises(FeatureNotFoundException, self.cube.get_second("Z"))

    def test_copy(self):
        copy = self.cube.copy()
        self.assertEqual(self.cube.dimensions, copy.dimensions)
        self.assertEqual(self.cube.mean, copy.mean)

    def test_expand_all(self):
        updates = [MinUpdate("X", 0.2), MinUpdate("Y", 0.15)]
        surrounding = TestUtility.create_cube((0.0, 0.8), (0.1, 0.6), 0.0)
        cube = TestUtility.create_cube((0.1, 0.2), (0.4, 0.4), 0.4)
        cube.expand_all(updates, surrounding)
        self.assertEqual((0.0, 0.4), cube.dimensions["X"])
        self.assertEqual((0.25, 0.55), cube.dimensions["Y"])

    def test_overlap(self):
        cube = TestUtility.get_cube()
        hypercubes = TestUtility.get_cubes()
        self.assertIsNone(cube.overlap(hypercubes))
        self.assertFalse(cube.overlap(hypercubes[0]))
        self.assertFalse(cube.overlap(hypercubes[1]))
        cube.update_dimension("X", 0.6, 1.0)
        self.assertIsNotNone(cube.overlap(hypercubes))
        self.assertEqual(hypercubes[1], cube.overlap(hypercubes))
        self.assertFalse(cube.overlap(hypercubes[0]))
        self.assertTrue(cube.overlap(hypercubes[1]))

    def test_has_volume(self):
        self.assertTrue(self.cube.has_volume())
        no_volume = self.cube.copy()
        no_volume.update_dimension("X", 1.0, 1.0)
        self.assertFalse(no_volume.has_volume())

    def test_equal(self):
        self.assertTrue(self.cube == self.cube)
        self.assertFalse(self.cube.equal(self.hypercubes))
        self.assertTrue(self.hypercubes[0].equal(self.hypercubes))

    def test_count(self):
        self.assertEqual(self.dataset.shape[0], HyperCube.create_surrounding_cube(self.dataset).count(self.dataset))
        self.assertEqual(self.filtered_dataset.shape[0], self.cube.count(self.dataset))

    def test_create_tuple(self):
        point = self.cube.create_tuple()
        for k, v in self.cube.dimensions.items():
            self.assertTrue(v[0] <= point[k])
            self.assertTrue(point[k] < v[1])

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


if __name__ == '__main__':
    unittest.main()
