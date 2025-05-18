from functools import lru_cache
from psyke.extraction import PedagogicalExtractor
from psyke.extraction.real.utils import Rule, IndexedRuleSet
from psyke.schema import DiscreteFeature
from psyke.utils.dataframe import HashableDataFrame
from psyke.utils.logic import create_term, create_head, create_variable_list
from tuprolog.core import Var, Struct, Clause, clause
from tuprolog.theory import MutableTheory, mutable_theory, Theory
from typing import Iterable
import pandas as pd
import numpy as np


class REAL(PedagogicalExtractor):
    """
    Explanator implementing Rule Extraction As Learning (REAL) algorithm, doi:10.1016/B978-1-55860-335-6.50013-1.
    The algorithm is sensible to features' order in the provided dataset during extraction.
    """

    def __init__(self, predictor, discretization: Iterable[DiscreteFeature]):
        super().__init__(predictor, discretization)
        self._ignore_feature = []
        self._ruleset: IndexedRuleSet = IndexedRuleSet()

    @property
    def n_rules(self):
        return len(self._ruleset.flatten())

    def _covers(self, sample: pd.Series, rules: list[Rule]) -> bool:
        new_rule = self._rule_from_example(sample)
        return any([new_rule in rule for rule in rules])

    def _body(self, variables: dict[str, Var], rule: Rule) -> list[Struct]:
        result = []
        for predicates, truth_value in zip(rule.to_lists(), [True, False]):
            for predicate in predicates:
                feature = [feature for feature in self.discretization if predicate in feature.admissible_values][0]
                result.append(create_term(variables[feature.name], feature.admissible_values[predicate], truth_value))
        return result

    def _create_clause(self, dataset: pd.DataFrame, variables: dict[str, Var], key: int, rule: Rule) -> Clause:
        return clause(create_head(dataset.columns[-1], list(variables.values()), key), self._body(variables, rule))

    def _create_new_rule(self, sample: pd.Series) -> Rule:
        rule = self._rule_from_example(sample)
        return self._generalise(rule, sample)

    def _create_ruleset(self, dataset: pd.DataFrame) -> IndexedRuleSet:
        ruleset = IndexedRuleSet.create_indexed_ruleset(sorted(set(dataset.iloc[:, -1])))
        for _, sample in dataset.iloc[:, :-1].iterrows():
            prediction = list(self.predictor.predict(sample.to_frame().transpose()))[0]
            rules = ruleset.get(prediction)
            if not self._covers(sample, rules):
                rules.append(self._create_new_rule(sample))
        return ruleset.optimize()

    def _create_theory(self, dataset: pd.DataFrame) -> MutableTheory:
        theory = mutable_theory()
        for key, rule in self._ruleset.flatten():
            variables = create_variable_list(self.discretization)
            theory.assertZ(self._create_clause(dataset, variables, key, rule))
        return theory

    def _generalise(self, rule: Rule, sample: pd.Series) -> Rule:
        mutable_rule = rule.to_lists()
        samples = sample.to_frame().transpose()
        for predicate in rule.true_predicates:
            samples = self._remove_antecedent(samples.copy(), predicate, mutable_rule)
        return Rule(mutable_rule[0], mutable_rule[1]).reduce(self.discretization)

    def _remove_antecedent(self, samples: pd.DataFrame, predicate: str, rule: list[list[str]]) -> (pd.DataFrame, bool):
        feature = [feature for feature in self.discretization if predicate in feature.admissible_values][0]
        output = np.array(self.predictor.predict(samples))
        copies = [samples.copy()]
        samples[predicate] = 0
        for f in [f for f in feature.admissible_values if f != predicate]:
            copy = samples.copy()
            copy[f] = 1
            if all(output == np.array(self.predictor.predict(copy))):
                copies.append(copy)
                rule[1].remove(f)
        if len(copies) > 1:
            rule[0].remove(predicate)
        return pd.concat([df for df in copies], ignore_index=True)

    @lru_cache(maxsize=512)
    def _get_or_set(self, dataset: HashableDataFrame) -> IndexedRuleSet:
        return self._create_ruleset(dataset)

    def _internal_predict(self, sample: pd.Series):
        x = [index for index, rule in self._ruleset.flatten() if self._rule_from_example(sample) in rule]
        return x[0] if x else None

    def make_fair(self, features: Iterable[str]):
        self._ignore_feature = [list(i.admissible_values.keys()) for i in self.discretization if i.name in features] \
            if self.discretization else [features]
        self._ignore_feature = [feature for features in self._ignore_feature for feature in features]
        self._get_or_set.cache_clear()

    def _rule_from_example(self, sample: pd.Series) -> Rule:
        true_predicates, false_predicates = [], []
        for feature, value in sample.items():
            if feature in self._ignore_feature:
                continue
            true_predicates.append(str(feature)) if value == 1 else false_predicates.append(str(feature))
        return Rule(true_predicates, false_predicates)

    def _subset(self, samples: pd.DataFrame, predicate: str) -> (pd.DataFrame, bool):
        samples_0 = samples.copy()
        samples_0[predicate].values[:] = 0
        samples_1 = samples.copy()
        samples_1[predicate].values[:] = 1
        samples_all = samples_0.append(samples_1)
        return samples_all, len(set(self.predictor.predict(samples_all))) == 1

    def _extract(self, dataframe: pd.DataFrame) -> Theory:
        self._ruleset = self._get_or_set(HashableDataFrame(dataframe))
        return self._create_theory(dataframe)

    def _predict(self, dataframe) -> Iterable:
        return np.array([self._internal_predict(data.transpose()) for _, data in dataframe.iterrows()])
