from __future__ import annotations
import pandas as pd

from psyke.classification.real.rule import Rule


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
