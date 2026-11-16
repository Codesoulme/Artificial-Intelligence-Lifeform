"""Integrity checks for the fixed, manually authored diagnostic set."""
import unittest
from collections import Counter

from scripts.deepseek_spora_event_scope_cross_domain_20261117 import CASES
from scripts.deepseek_spora_event_scope_probe_20261117 import LABELS, NAMES, TEMPLATES


class CrossDomainSetTests(unittest.TestCase):
    def test_balanced_unique_and_disjoint_from_prior_questions(self):
        self.assertEqual(Counter(label for label, _, _ in CASES), {label: 6 for label in LABELS})
        questions = {question for _, _, question in CASES}
        self.assertEqual(len(questions), len(CASES))
        previous = {template.format(n=name, name=name)
                    for part, names in NAMES.items() for name in names
                    for templates in TEMPLATES[part].values() for template in templates}
        self.assertTrue(questions.isdisjoint(previous))

    def test_each_positive_has_six_different_topics(self):
        for label in LABELS[:3]:
            self.assertEqual(len({domain for kind, domain, _ in CASES if kind == label}), 6)
