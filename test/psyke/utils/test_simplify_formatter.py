import unittest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from psyke import Extractor, get_default_random_seed
from psyke.extraction.hypercubic import Grid
from test import get_dataset


class TestSimplifyFormatter(unittest.TestCase):

    def test_simplify_formatter(self):
        iris_data = get_dataset('house')
        train, test = train_test_split(iris_data, test_size=0.5, random_state=get_default_random_seed())
        predictor = DecisionTreeRegressor()
        predictor.fit(train.iloc[:, :-1], train.iloc[:, -1])
        extractor = Extractor.gridrex(predictor, Grid())
        theory = extractor.extract(train)
        # print(pretty_theory(theory))


if __name__ == '__main__':
    unittest.main()
