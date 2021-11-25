from __future__ import annotations
from psyke import DiscreteFeature
from typing import Iterable
import pandas as pd


class Rule:

    def __init__(self, true_predicates: list[str], false_predicates: list[str]):
        self.true_predicates = true_predicates
        self.false_predicates = false_predicates

    def is_sub_rule_of(self, rule: Rule) -> bool:
        return all([predicate in self.true_predicates for predicate in rule.true_predicates]) & \
               all([predicate in self.false_predicates for predicate in rule.false_predicates])

    def reduce(self, features: Iterable[DiscreteFeature]) -> Rule:
        to_be_removed = [feature.admissible_values.keys() for tp in self.true_predicates
                         for feature in features if tp in feature.admissible_values]
        to_be_removed = [item for sublist in to_be_removed for item in sublist]
        return Rule(self.true_predicates, [fp for fp in self.false_predicates if fp not in to_be_removed])

    def to_lists(self) -> list[list[str]]:
        return [self.true_predicates.copy(), self.false_predicates.copy()]


class IndexedRuleSet(dict[int, list[Rule]]):

    def flatten(self) -> list[tuple[int, Rule]]:
        return [(key, value) for key, values in self.items() for value in values]

    def optimize(self) -> IndexedRuleSet:
        useless_rules = [IndexedRuleSet._useless_rules(key, entry) for key, entry in self.items()]
        useless_rules = [] if len(useless_rules) == 0 else [item for sublist in useless_rules for item in sublist]
        for rule in useless_rules:
            self[rule[0]].remove(rule[1])
        return self

    @staticmethod
    def _useless_rules(key, rules: list[Rule]) -> list[(int, Rule)]:
        result = []
        if len(rules) > 1:
            for rule in rules:
                append = True
                for other_rule in rules:
                    if (rule != other_rule) and not rule.is_sub_rule_of(other_rule):
                        append = False
                        break
                if append:
                    result.append((key, rule))
        return result
        # return [(key, rule) for rule in rules if not any(rule.is_sub_rule_of(aux) for aux in rules if aux != rule)]

    @staticmethod
    def create_indexed_ruleset(dataset: pd.DataFrame) -> IndexedRuleSet:
        return IndexedRuleSet({index: [] for index, _ in enumerate(set(dataset.iloc[:, -1]))})