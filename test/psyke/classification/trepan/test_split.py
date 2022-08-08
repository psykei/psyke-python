from psyke.extraction.trepan import Node, Split
from test import get_dataset
import math
import pandas as pd
import unittest


class TestSplit(unittest.TestCase):

    dataset: pd.DataFrame = get_dataset('iris')
    n_examples = dataset.shape[0]
    all_node = Node(dataset, n_examples)
    setosa_40 = Node(dataset.iloc[10:70, :], n_examples)
    setosa_40_complementar = Node(dataset.iloc[:10, :].append(dataset.iloc[70:, :]), n_examples)
    versicolor_25 = Node(dataset.iloc[40:75, :], n_examples)
    versicolor_25_complementar = Node(dataset.iloc[75:110, :], n_examples)

    def test_priority(self):
        self.assertTrue(math.isclose(-40/60-50/90-100,
                                     Split(self.all_node, (self.setosa_40, self.setosa_40_complementar)).priority))
        self.assertTrue(math.isclose((25 / 35) * - 2 - 200 + 200,
                                     Split(self.all_node, (self.versicolor_25, self.versicolor_25_complementar))
                                     .priority))


if __name__ == '__main__':
    unittest.main()
