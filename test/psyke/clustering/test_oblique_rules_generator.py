import unittest
from psyke.extraction.hypercubic.orbit.oblique_rules_generator import *
from psyke.extraction.hypercubic.hypercube import ClosedCube


class AbstractTestObliqueRulesGenerator(unittest.TestCase):
    def setUp(self):
        self.contour_net = {(0, 0): ((0, 1), (1, 0)),
                            (1, 0): ((0, 0), (2, 1)),
                            (0, 1): ((0, 0), (1, 2)),
                            (2, 1): ((1, 2), (1, 0)),
                            (1, 2): ((0, 1), (2, 1))}
        self.elimination_cost = {(0, 0): 1,
                                 (1, 0): 1,
                                 (0, 1): 1,
                                 (2, 1): 0.5,
                                 (1, 2): 1}
        self.contour_net_square = {(0, 0): ((0, 1), (1, 0)),
                                   (1, 0): ((0, 0), (1, 1)),
                                   (0, 1): ((0, 0), (1, 1)),
                                   (1, 1): ((1, 0), (1, 0))}
        self.hull_lines = [
            [(0, 0), (0, 1)],
            [(0, 1), (1, 1)],
            [(1, 1), (1, 0)],
            [(1, 0), (0, 0)]
        ]
        self.df = pd.DataFrame([[0, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [2, 1, 0],
                                [1, 2, 0],
                                [1, 1, 0]],
                               columns=["X", "Y", "cluster"])
        self.iper_cube = ClosedCube(dimension={"X": (0, 10),
                                               "Y": (0, 10)})


class TestObliqueRulesGenerator(AbstractTestObliqueRulesGenerator):
    def test_generate_container(self):
        container = generate_container(self.df, self.df, [0]*self.df.shape[0], self.iper_cube, 100, 0, 10)
        self.assertGreater(len(container.dimensions) + len(container.inequalities), 0)

    def test_extract_points(self):
        points = extract_points(self.contour_net)
        for point in self.contour_net:
            self.assertIn(point, points)

    def simplify_convex_hull(self):
        contour_net = simplify_convex_hull(self.hull_lines)
        self.assertGreater(len(contour_net), 2)

    def test_generate_contour_net(self):
        contour_net = generate_contour_net(self.hull_lines)
        self.assertEqual(4, len(contour_net))

    def test_generate_disequations(self):
        disequations = generate_disequations(self.contour_net)
        self.assertEqual(len(self.contour_net), len(disequations))

    def test_get_disequation(self):
        a, b, c = get_disequation((0, 0), (1, 1), (1, 0))
        self.assertEqual(0, round(c, 3))
        self.assertEqual(-1, round(b / a, 3))
        self.assertGreater(b, 0)

    def test_evaluate_elimination_cost(self):
        c_n = self.contour_net.copy()
        e_c = evaluate_elimination_cost((2, 1), c_n)
        self.assertEqual(c_n, self.contour_net)
        self.assertIsInstance(e_c, float)

    def test_eliminate_point(self):
        c_n = self.contour_net.copy()
        e_c = self.elimination_cost.copy()
        eliminate_point((2, 1), c_n, e_c)
        self.assertEqual(4, len(c_n))
        self.assertEqual(4, len(e_c))

    def test_get_new_points(self):
        p = list(self.contour_net.keys())[0]
        new_p0, new_p1, e_a0, e_a1, e_p0, e_p1, p0, p1 = get_new_points(p, self.contour_net)
        self.assertIsInstance(new_p0, tuple)
        self.assertIsInstance(new_p1, tuple)
        self.assertIsInstance(e_a0, float)
        self.assertIsInstance(e_a1, float)
        self.assertIsInstance(e_p0, tuple)
        self.assertIsInstance(e_p1, tuple)
        self.assertIsInstance(p0, tuple)
        self.assertIsInstance(p1, tuple)
        p = list(self.contour_net_square.keys())[0]
        new_p0, new_p1, e_a0, e_a1, e_p0, e_p1, p0, p1 = get_new_points(p, self.contour_net_square)
        self.assertEqual(new_p0, None)
        self.assertEqual(new_p1, None)
        self.assertEqual(e_a0, np.inf)
        self.assertEqual(e_a1, np.inf)

    def test_get_rect(self):
        a, b, c = get_rect((0, 1), (1, 2))
        self.assertEqual((-1, 1, 1), (round(a / c, 3), round(b / c, 3), 1))

        a, b, c = get_rect((0, 1), (2, 1))                  # horizontal line
        self.assertEqual((1, 1), (round(b / c, 3), 1))      # check b, c
        self.assertEqual(0, a)

        a, b, c = get_rect((1, 0), (1, 2))                  # vertical line
        self.assertEqual((1, 1), (round(a / c, 3), 1))      # check a, c
        self.assertEqual(0, b)

    def test_get_intersection(self):
        self.assertEqual((1, 1), get_intersection((1, 1, 2), (1, -1, 0)))
        self.assertEqual((1, 1), get_intersection((1, 0, 1), (0, 1, 1)))

    def test_get_area(self):
        self.assertEqual(1, get_area((0, 0), (2, 0), (1, 1)))
        self.assertEqual(3, get_area((1, 1), (1, 3), (4, 3)))

    def test_is_middle_point(self):
        self.assertTrue(is_middle_point((1, 1), (0, 0), (2, 2)))
        self.assertFalse(is_middle_point((0, 0), (1, 1), (2, 2)))

    def test_get_total_accuracy(self):
        self.assertEqual(0.8, get_total_accuracy(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]), 10))
