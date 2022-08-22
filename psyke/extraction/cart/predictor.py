from collections import Iterable
from typing import Union, Any
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from psyke.schema import Value, LessThan, GreaterThan

LeafConstraints = list[tuple[str, Value, bool]]
LeafSequence = Iterable[tuple[LeafConstraints, Any]]


class CartPredictor:
    """
    A wrapper for decision and regression trees of sklearn.
    """

    def __init__(self, predictor: Union[DecisionTreeClassifier, DecisionTreeRegressor] = DecisionTreeClassifier(),
                 normalization=None):
        self._predictor = predictor
        self.normalization = normalization

    def __get_constraints(self, nodes: Iterable[(int, bool)]) -> LeafConstraints:
        thresholds = [self._predictor.tree_.threshold[i[0]] for i in nodes]
        features = [self._predictor.feature_names_in_[self._predictor.tree_.feature[node[0]]] for node in nodes]
        conditions = [node[1] for node in nodes]
        if self.normalization is not None:
            thresholds = [threshold * self.normalization[feature][1] + self.normalization[feature][0]
                          for feature, threshold in zip(features, thresholds)]
        return [(feature, LessThan(threshold) if condition else GreaterThan(threshold), condition)
                for feature, condition, threshold in zip(features, conditions, thresholds)]

    def __get_leaves(self) -> Iterable[int]:
        return [i for i, (left_child, right_child) in enumerate(zip(
            self._left_children, self._right_children
        )) if left_child == -1 and right_child == -1]

    def __get_prediction(self, node: int) -> Any:
        if hasattr(self._predictor, 'classes_'):
            return self._predictor.classes_[np.argmax(self._predictor.tree_.value[node])]
        else:
            return self._predictor.tree_.value[node]

    def __path(self, node: int, path=[]) -> Iterable[(int, bool)]:
        if node == 0:
            return path
        father = list(self._left_children if node in self._left_children else self._right_children).index(node)
        return self.__path(father, [(father, node in self._left_children)] + path)

    def __iter__(self) -> LeafSequence:
        leaves = self.__get_leaves()
        return ((self.__get_constraints(self.__path(i)), self.__get_prediction(i)) for i in leaves)

    def predict(self, data) -> Iterable:
        return self._predictor.predict(data)

    @property
    def predictor(self) -> Union[DecisionTreeClassifier, DecisionTreeRegressor]:
        return self._predictor

    @property
    def n_leaves(self) -> int:
        return len(list(self.__get_leaves()))

    @property
    def _left_children(self) -> list[int]:
        return self._predictor.tree_.children_left

    @property
    def _right_children(self) -> list[int]:
        return self._predictor.tree_.children_right

    @predictor.setter
    def predictor(self, predictor: Union[DecisionTreeClassifier, DecisionTreeRegressor]):
        self._predictor = predictor
