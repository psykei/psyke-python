from collections import Iterable
from typing import Union, Any
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from psyke.cart.cart_utils import LeafSequence, LeafConstraints
from psyke.schema.value import LessThan, GreaterThan
from psyke.utils import get_default_random_seed


class CartPredictor:
    """
    Wrapper for decision trees and regression trees of sklearn.
    """

    def __init__(self, predictor: Union[DecisionTreeClassifier, DecisionTreeRegressor] = None):
        self.predictor = predictor

    def reset_regression_tree(self):
        self.predictor = DecisionTreeRegressor(random_state=get_default_random_seed())

    def reset_classifier_tree(self):
        self.predictor = DecisionTreeClassifier(random_state=get_default_random_seed())

    def predict(self, data) -> Iterable:
        return self.predictor.predict(data)

    def as_sequence(self) -> LeafSequence:
        leaves = self._get_leaves()
        return [(self._get_constraints(self._path(i)), self._get_prediction(i)) for i in leaves]

    def _get_leaves(self) -> Iterable[int]:
        left_orphan = [i for i, v in enumerate(self.predictor.tree_.children_left) if v == -1]
        right_orphan = [i for i, v in enumerate(self.predictor.tree_.children_right) if v == -1]
        return [v for v in left_orphan if v in left_orphan and v in right_orphan]

    def _path(self, node: int, path=[]) -> Iterable[(int, bool)]:
        if node == 0:
            return path
        else:
            father_left = [(i, True) for i, v in enumerate(self.predictor.tree_.children_left) if v == node]
            father_right = [(i, False) for i, v in enumerate(self.predictor.tree_.children_right) if v == node]
            father: (int, bool) = (father_left + father_right)[0]
            return self._path(father[0], [father] + path)

    def _get_constraints(self, nodes: Iterable[(int, bool)]) -> LeafConstraints:
        return [(self.predictor.feature_names_in_[self.predictor.tree_.feature[i[0]]],
                 LessThan(self.predictor.tree_.threshold[i[0]]) if i[1] else
                 GreaterThan(self.predictor.tree_.threshold[i[0]])) for i in nodes]

    def _get_prediction(self, node: int) -> Any:
        return self.predictor.classes_[np.argmax(self.predictor.tree_.value[node])]
