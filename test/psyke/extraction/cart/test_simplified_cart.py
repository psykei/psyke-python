from typing import Callable

import numpy as np
from parameterized import parameterized_class
from sklearn.model_selection import train_test_split
from tuprolog.solve.prolog import prolog_solver
from tuprolog.theory import mutable_theory

from psyke import Extractor
from psyke.utils import get_default_precision
from psyke.utils.logic import data_to_struct, get_in_rule, get_not_in_rule
from test import get_dataset, get_model
import unittest


# TODO: should be refactored using the a .csv file
from test.psyke import get_substitutions


@parameterized_class([{"dataset": "iris", "predictor": "DTC", "task": "extraction"},
                      {"dataset": "house", "predictor": "DTR", "task": "hypercubic"}])
class TestSimplifiedCart(unittest.TestCase):

    def test_equality(self):
        dataset = get_dataset(self.dataset)
        dataset = dataset.reindex(sorted(dataset.columns[:-1]) + [dataset.columns[-1]], axis=1)
        train, test = train_test_split(dataset, test_size=0.5)
        tree, _ = get_model(self.predictor, {})
        tree.fit(train.iloc[:, :-1], train.iloc[:, -1])
        extractor = Extractor.cart(tree, simplify=False)
        theory = extractor.extract(train)
        simplified_extractor = Extractor.cart(tree)
        simplified_theory = simplified_extractor.extract(train)

        index = test.shape[1] - 1
        cast, substitutions = get_substitutions(test, test, theory)
        expected = [cast(query.solved_query.get_arg_at(index)) for query in substitutions]

        cast, simplified_substitutions = get_substitutions(test, test, simplified_theory)
        simplified_expected = [cast(query.solved_query.get_arg_at(index)) for query in simplified_substitutions]

        if isinstance(test.iloc[0, -1], str):
            self.assertTrue(all(np.array(extractor.predict(test.iloc[:, :-1])) ==
                                np.array(simplified_extractor.predict(test.iloc[:, :-1]))))
            self.assertEqual(expected, simplified_expected)
        else:
            self.assertTrue(max(abs(np.array(extractor.predict(test.iloc[:, :-1])) -
                                    np.array(simplified_extractor.predict(test.iloc[:, :-1])))
                                ) < get_default_precision())
            self.assertTrue(max(abs(np.array(expected) - np.array(simplified_expected))) < get_default_precision())


if __name__ == '__main__':
    unittest.main()
