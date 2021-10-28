import unittest

from psyke.regression.hypercube import HyperCube


class CubeBuilder:

    def __init__(self):
        self.cube = HyperCube()

    def update_dimension(self, feature, values):
        self.cube.update_dimension(feature, values)
        return self


class TestOverlap(unittest.TestCase):

    def test_no_overlap(self):
        cube_one = CubeBuilder().update_dimension('X', (-5, 5)).update_dimension('Y', (7, 10)).cube
        cube_two = CubeBuilder().update_dimension('X', (6, 14)).update_dimension('Y', (5, 6)).cube

        self.assertFalse(cube_one.overlap(cube_two))

    def test_overlap(self):
        cube_one = CubeBuilder().update_dimension('X', (-5, 5)).update_dimension('Y', (7, 10)).cube
        cube_two = CubeBuilder().update_dimension('X', (7, 12)).update_dimension('Y', (-3, 8)).cube

        self.assertTrue(cube_one.overlap(cube_two))


class TestOverlapAll(unittest.TestCase):

    def test_no_overlap_all(self):
        cube_one = CubeBuilder().update_dimension('X', (-2, 5)).update_dimension('Y', (7, 10)).cube
        cube_two = CubeBuilder().update_dimension('X', (6, 8)).update_dimension('Y', (5, 6)).cube
        cube_three = CubeBuilder().update_dimension('X', (10, 14)).update_dimension('Y', (0, 3)).cube

        self.assertFalse(cube_one.overlap_all((cube_two, cube_three)))

    def test_overlap_all(self):
        cube_one = CubeBuilder().update_dimension('X', (-5, 5)).update_dimension('Y', (7, 10)).cube
        cube_two = CubeBuilder().update_dimension('X', (7, 12)).update_dimension('Y', (-3, 8)).cube
        cube_three = CubeBuilder().update_dimension('X', (10, 14)).update_dimension('Y', (0, 3)).cube

        self.assertTrue(cube_one.overlap_all((cube_two, cube_three)))


if __name__ == '__main__':
    unittest.main()
