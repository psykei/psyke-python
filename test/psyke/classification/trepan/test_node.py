from psyke.extraction.trepan import Node
from test import get_dataset
import pandas as pd
import unittest


class TestNode(unittest.TestCase):

    dataset: pd.DataFrame = get_dataset('iris')
    n_examples = dataset.shape[0]
    all_node = Node(dataset, n_examples)
    setosa_40 = Node(dataset.iloc[10:70, :], n_examples)
    versicolor_10 = Node(dataset.iloc[95:110, :], n_examples)
    virginica_50 = Node(dataset.iloc[20:130, :], n_examples)

    def test_reach(self):
        node = Node(self.dataset, self.n_examples)
        self.assertEqual(node.reach, self.all_node.reach)
        self.assertTrue(self.versicolor_10.reach < self.setosa_40.reach)
        self.assertTrue(self.setosa_40.reach < self.virginica_50.reach)
        self.assertTrue(self.virginica_50.reach < self.all_node.reach)

    def test_dominant(self):
        self.assertEqual('setosa', self.setosa_40.dominant)
        self.assertEqual('virginica', self.virginica_50.dominant)
        self.assertEqual('versicolor', self.versicolor_10.dominant)

    def test_correct(self):
        self.assertEqual(50, self.virginica_50.correct)
        self.assertEqual(40, self.setosa_40.correct)
        self.assertEqual(10, self.versicolor_10.correct)

    def test_fidelity(self):
        self.assertEqual(50 / 150, self.all_node.fidelity)
        self.assertEqual(40 / 60, self.setosa_40.fidelity)
        self.assertEqual(10 / 15, self.versicolor_10.fidelity)
        self.assertEqual(50 / 110, self.virginica_50.fidelity)

    def test_priority(self):
        self.assertTrue(self.all_node.priority < self.virginica_50.priority)
        self.assertTrue(self.virginica_50.priority < self.setosa_40.priority)
        self.assertTrue(self.setosa_40.priority < self.versicolor_10.priority)

    def test_n_classes(self):
        self.assertEqual(3, self.all_node.n_classes)
        self.assertEqual(2, self.versicolor_10.n_classes)
        self.assertEqual(2, self.setosa_40.n_classes)
        self.assertEqual(3, self.virginica_50.n_classes)
        self.assertEqual(1, Node(self.dataset.iloc[15:40, :], self.n_examples).n_classes)

    def test_iterator(self):
        node = Node(self.dataset, self.n_examples)
        child_1 = Node(self.dataset.iloc[:50, :], self.n_examples)
        child_2 = Node(self.dataset.iloc[50:150, :], self.n_examples)
        node.children = [child_1, child_2]
        grandchild_1_1 = Node(self.dataset.iloc[:25, :], self.n_examples)
        grandchild_2_1 = Node(self.dataset.iloc[50:80, :], self.n_examples)
        grandchild_2_2 = Node(self.dataset.iloc[80:120, :], self.n_examples)
        child_1.children = [grandchild_1_1]
        child_2.children = [grandchild_2_1, grandchild_2_2]
        self.assertEqual(list(node), list(child_1) + list(child_2) + [node])
        self.assertEqual([grandchild_1_1, child_1, grandchild_2_1, grandchild_2_2, child_2, node], list(node))

    def test_to_string(self):
        node = Node(self.dataset, self.n_examples, (('V1', 0.0), ('V2', 1.0)))
        self.assertEqual(' = setosa', str(self.all_node))
        self.assertEqual(' = setosa', str(self.setosa_40))
        self.assertEqual(' = virginica', str(self.virginica_50))
        self.assertEqual(' = versicolor', str(self.versicolor_10))
        self.assertEqual('!V1, V2 = setosa', str(node))


if __name__ == '__main__':
    unittest.main()
