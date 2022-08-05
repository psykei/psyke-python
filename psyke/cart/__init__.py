from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from psyke.cart.predictor import CartPredictor, LeafConstraints, LeafSequence
from psyke import Extractor, DiscreteFeature, get_default_random_seed
from psyke.schema import GreaterThan
from psyke.utils.logic import create_variable_list, create_head, create_term
from tuprolog.core import clause, Var, Struct
from tuprolog.theory import Theory, mutable_theory
from typing import Iterable
import pandas as pd


TREE_SEED = get_default_random_seed()


class Cart(Extractor):

    def __init__(self, predictor, max_depth: int = 3, max_leaves: int = None,
                 discretization: Iterable[DiscreteFeature] = None, simplify: bool = True):
        super().__init__(predictor, discretization)
        self._cart_predictor = CartPredictor()
        self.depth = max_depth
        self.leaves = max_leaves
        self._simplify = simplify

    def _create_body(self, variables: dict[str, Var], constraints: LeafConstraints) -> Iterable[Struct]:
        results = []
        for name, value in constraints:
            features = [d for d in self.discretization if name in d.admissible_values]
            feature: DiscreteFeature = features[0] if len(features) > 0 else None
            if feature is None:
                results.append(create_term(variables[name], value))
            else:
                results.append(create_term(
                    variables[feature.name],
                    feature.admissible_values[name],
                    isinstance(value, GreaterThan))
                )
        return results

    def _create_theory(self, data: pd.DataFrame) -> Theory:
        new_theory = mutable_theory()
        for name, value in self._cart_predictor:
            # TODO: probably there is a bug in simplify.
            name = [(n[0], n[1]) for n in name if not self._simplify or n[2]]
            variables = create_variable_list(self.discretization, data)
            new_theory.assertZ(
                clause(
                    create_head(data.columns[-1], list(variables.values()), value),
                    self._create_body(variables, name)
                )
            )
        return new_theory

    def extract(self, data: pd.DataFrame) -> Theory:
        self._cart_predictor.predictor = DecisionTreeClassifier(random_state=TREE_SEED) \
            if isinstance(data.iloc[0, -1], str) else DecisionTreeRegressor(random_state=TREE_SEED)
        self._cart_predictor.predictor.max_depth = self.depth
        self._cart_predictor.predictor.max_leaf_nodes = self.leaves
        self._cart_predictor.predictor.fit(data.iloc[:, :-1], self.predictor.predict(data.iloc[:, :-1]))
        return self._create_theory(data)

    def predict(self, data) -> Iterable:
        return self._cart_predictor.predict(data)

    @property
    def n_rules(self) -> int:
        return self._cart_predictor.n_leaves
