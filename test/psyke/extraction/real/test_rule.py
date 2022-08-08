import unittest
from psyke.extraction.real.utils import Rule
from psyke.utils.dataframe import split_features
from test import get_dataset


class TestRule(unittest.TestCase):

    def test_subrule(self):
        pred_1, pred_2 = ['V1', 'V2'], ['V3', 'V4']
        rule_1 = Rule(pred_1, pred_2)
        self.assertTrue(rule_1 in rule_1)
        rule_2 = Rule(pred_2, pred_1)
        self.assertFalse(rule_1 in rule_2)
        self.assertFalse(rule_2 in rule_1)
        rule_3 = Rule(['V1'], ['V3'])
        self.assertTrue(rule_1 in rule_3)
        self.assertFalse(rule_3 in rule_1)
        self.assertFalse(rule_2 in rule_3)
        self.assertFalse(rule_3 in rule_2)
        rule_4 = Rule(["V1"], ["V5"])
        self.assertFalse(rule_1 in rule_4)
        self.assertFalse(rule_4 in rule_1)
        rule_5 = Rule(["V1", "V6"], ["V3", "V4"])
        self.assertFalse(rule_1 in rule_5)
        self.assertFalse(rule_5 in rule_1)
        self.assertTrue(rule_1 in Rule([], []))

    def test_reduce(self):
        dataset = get_dataset('iris')
        features = split_features(dataset)
        rule = Rule(["V1_1", "V2_2", "V3_0"],
                    ["V1_0", "V2_1", "V2_0", "V4_1", "V4_2"])
        reduced_rule = Rule(["V1_1", "V2_2", "V3_0"],
                            ["V4_1", "V4_2"])
        self.assertEqual(reduced_rule.true_predicates, rule.reduce(features).true_predicates)
        self.assertEqual(reduced_rule.false_predicates, rule.reduce(features).false_predicates)
        self.assertEqual(reduced_rule.true_predicates, reduced_rule.reduce(features).true_predicates)
        self.assertEqual(reduced_rule.false_predicates, reduced_rule.reduce(features).false_predicates)


if __name__ == '__main__':
    unittest.main()