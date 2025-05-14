from __future__ import annotations
from psyke import DiscreteFeature
from typing import Iterable
import pandas as pd


class Rule:

    def __init__(self, true_predicates: list[str], false_predicates: list[str]):
        self.true_predicates = true_predicates
        self.false_predicates = false_predicates

    def __contains__(self, other: Rule) -> bool:
        return all([predicate in other.true_predicates for predicate in self.true_predicates]) and\
               all([predicate in other.false_predicates for predicate in self.false_predicates])

    def __eq__(self, other: Rule) -> bool:
        return self.true_predicates == other.true_predicates and self.false_predicates == other.false_predicates

    def __hash__(self) -> int:
        return hash(self.true_predicates) + hash(self.false_predicates)

    def reduce(self, features: Iterable[DiscreteFeature]) -> Rule:
        to_be_removed = [item for tp in self.true_predicates
                         for feature in features if tp in feature.admissible_values
                         for item in feature.admissible_values.keys()]
        return Rule(self.true_predicates, [fp for fp in self.false_predicates if fp not in to_be_removed])

    def to_lists(self) -> list[list[str]]:
        return [self.true_predicates.copy(), self.false_predicates.copy()]


class IndexedRuleSet(dict[int, list[Rule]]):

    def flatten(self) -> list[tuple[int, Rule]]:
        return [(key, value) for key, values in self.items() for value in values]

    def optimize(self) -> IndexedRuleSet:
        useless_rules = [item for key, entry in self.items() for item in IndexedRuleSet._useless_rules(key, entry)]
        for rule in useless_rules:
            self[rule[0]].remove(rule[1])
        return self

    @staticmethod
    def _useless_rules(key, rules: list[Rule]) -> list[(int, Rule)]:
        return [
            (key, rule) for rule in rules
            if any(rule in other_rule for other_rule in rules if other_rule != rule)
        ]

    @staticmethod
    def create_indexed_ruleset(indices: Iterable) -> IndexedRuleSet:
        return IndexedRuleSet({i: [] for i in indices})
