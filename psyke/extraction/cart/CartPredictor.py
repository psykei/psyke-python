from collections.abc import Iterable
from typing import Union, Any
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tuprolog.core import clause, Var, Struct
from tuprolog.theory import Theory, mutable_theory

from psyke.extraction.cart import LeafConstraints, LeafSequence
from psyke.schema import LessThan, GreaterThan, SchemaException, DiscreteFeature
from psyke.utils.logic import create_variable_list, create_head, create_term


class CartPredictor:
    """
    A wrapper for decision and regression trees of sklearn.
    """

    def __init__(self, predictor: Union[DecisionTreeClassifier, DecisionTreeRegressor] = DecisionTreeClassifier(),
                 discretization=None, normalization=None):
        self._predictor = predictor
        self.discretization = discretization
        self.normalization = normalization

    def __get_constraints(self, nodes: Iterable[tuple[int, bool]]) -> LeafConstraints:
        thresholds = [self._predictor.tree_.threshold[i[0]] for i in nodes]
        features = [self._predictor.feature_names_in_[self._predictor.tree_.feature[node[0]]] for node in nodes]
        conditions = [node[1] for node in nodes]
        if self.normalization is not None:
            thresholds = [threshold * self.normalization[feature][1] + self.normalization[feature][0]
                          for feature, threshold in zip(features, thresholds)]
        cond_dict = {}
        for feature, condition, threshold in zip(features, conditions, thresholds):
            cond = LessThan(threshold) if condition else GreaterThan(threshold)
            if feature in cond_dict:
                try:
                    cond_dict[feature][-1] *= cond
                except SchemaException:
                    cond_dict[feature].append(cond)
            else:
                cond_dict[feature] = [cond]
        return cond_dict

    def __get_leaves(self) -> Iterable[int]:
        return [i for i, (left_child, right_child) in enumerate(zip(
            self._left_children, self._right_children
        )) if left_child == -1 and right_child == -1]

    def __get_prediction(self, node: int) -> Any:
        if hasattr(self._predictor, 'classes_'):
            return self._predictor.classes_[np.argmax(self._predictor.tree_.value[node])]
        else:
            return self._predictor.tree_.value[node]

    def __path(self, node: int, path=None) -> Iterable[tuple[int, bool]]:
        path = [] if path is None else path
        if node == 0:
            return path
        father = list(self._left_children if node in self._left_children else self._right_children).index(node)
        return self.__path(father, [(father, node in self._left_children)] + path)

    def __iter__(self) -> LeafSequence:
        leaves = self.__get_leaves()
        return ((self.__get_constraints(self.__path(i)), self.__get_prediction(i)) for i in leaves)

    def predict(self, data) -> Iterable:
        return self._predictor.predict(data)

    @staticmethod
    def _simplify_nodes(nodes: list) -> Iterable:
        simplified = [nodes.pop(0)]
        while len(nodes) > 0:
            first_node = nodes[0][0]
            for k, conditions in first_node.items():
                for condition in conditions:
                    if all(k in node[0] and condition in node[0][k] for node in nodes):
                        [node[0][k].remove(condition) for node in nodes]
            simplified.append(nodes.pop(0))
        return [({k: v for k, v in rule.items() if v != []}, prediction) for rule, prediction in simplified]

    def _create_body(self, variables: dict[str, Var], conditions: LeafConstraints) -> Iterable[Struct]:
        results = []
        for feature_name, cond_list in conditions.items():
            for condition in cond_list:
                feature: DiscreteFeature = [d for d in self.discretization if feature_name in d.admissible_values][0] \
                    if self.discretization else None
                results.append(create_term(variables[feature_name], condition) if feature is None else
                               create_term(variables[feature.name],
                                           feature.admissible_values[feature_name],
                                           isinstance(condition, GreaterThan)))
        return results

    def create_theory(self, data: pd.DataFrame, simplify: bool = True) -> Theory:
        new_theory = mutable_theory()
        nodes = [node for node in self]
        nodes = self._simplify_nodes(nodes) if simplify else nodes
        for (constraints, prediction) in nodes:
            if self.normalization is not None and data.columns[-1] in self.normalization:
                m, s = self.normalization[data.columns[-1]]
                prediction = prediction * s + m
            variables = create_variable_list(self.discretization, data)
            new_theory.assertZ(
                clause(
                    create_head(data.columns[-1], list(variables.values()), prediction),
                    self._create_body(variables, constraints)
                )
            )
        return new_theory

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
