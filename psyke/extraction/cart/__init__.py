from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from psyke.extraction.cart.predictor import CartPredictor, LeafConstraints, LeafSequence
from psyke import get_default_random_seed, PedagogicalExtractor
from psyke.schema import GreaterThan, DiscreteFeature
from psyke.utils.logic import create_variable_list, create_head, create_term
from tuprolog.core import clause, Var, Struct
from tuprolog.theory import Theory, mutable_theory
from typing import Iterable
import pandas as pd


TREE_SEED = get_default_random_seed()


class Cart(PedagogicalExtractor):

    def __init__(self, predictor, max_depth: int = 3, max_leaves: int = None,
                 discretization: Iterable[DiscreteFeature] = None,
                 normalization=None, simplify: bool = True):
        super().__init__(predictor, discretization, normalization)
        self._cart_predictor = CartPredictor(normalization=normalization)
        self.depth = max_depth
        self.leaves = max_leaves
        self._simplify = simplify

    def _create_body(self, variables: dict[str, Var], constraints: LeafConstraints) -> Iterable[Struct]:
        results = []
        for feature_name, constraint, value in constraints:
            features = [d for d in self.discretization if feature_name in d.admissible_values]
            feature: DiscreteFeature = features[0] if len(features) > 0 else None
            results.append(create_term(variables[feature_name], constraint) if feature is None else
                           create_term(variables[feature.name],
                                       feature.admissible_values[feature_name],
                                       isinstance(constraint, GreaterThan)))
        return results

    @staticmethod
    def _simplify_nodes(nodes: list) -> Iterable:
        simplified = [nodes.pop(0)]
        while len(nodes) > 0:
            first_node = nodes[0][0]
            for condition in first_node:
                if all([condition in [node[0] for node in nodes][i] for i in range(len(nodes))]):
                    [node[0].remove(condition) for node in nodes]
            simplified.append(nodes.pop(0))
        return simplified

    def _create_theory(self, data: pd.DataFrame, mapping: dict[str: int]) -> Theory:
        new_theory = mutable_theory()
        nodes = [node for node in self._cart_predictor]
        nodes = Cart._simplify_nodes(nodes) if self._simplify else nodes
        for (constraints, prediction) in nodes:
            if self.normalization is not None:
                m, s = self.normalization[data.columns[-1]]
                prediction = prediction * s + m
            if mapping is not None and prediction in mapping.values():
                for k, v in mapping.items():
                    if v == prediction:
                        prediction = k
                        break
            variables = create_variable_list(self.discretization, data)
            new_theory.assertZ(
                clause(
                    create_head(data.columns[-1], list(variables.values()), prediction),
                    self._create_body(variables, constraints)
                )
            )
        return new_theory

    def _extract(self, data: pd.DataFrame, mapping: dict[str: int] = None) -> Theory:
        self._cart_predictor.predictor = DecisionTreeClassifier(random_state=TREE_SEED) \
            if isinstance(data.iloc[0, -1], str) or mapping is not None else DecisionTreeRegressor(random_state=TREE_SEED)
        if mapping is not None:
            data.iloc[:, -1] = data.iloc[:, -1].apply(lambda x: mapping[x] if x in mapping.keys() else x)
        self._cart_predictor.predictor.max_depth = self.depth
        self._cart_predictor.predictor.max_leaf_nodes = self.leaves
        self._cart_predictor.predictor.fit(data.iloc[:, :-1], data.iloc[:, -1])
        return self._create_theory(data, mapping)

    def _predict(self, data) -> Iterable:
        return self._cart_predictor.predict(data)

    @property
    def n_rules(self) -> int:
        return self._cart_predictor.n_leaves
