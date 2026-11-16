"""Research probe invariants; these tests do not validate model generalization."""

import unittest

import torch

from scripts import deepseek_spora_event_scope_probe_20261117 as probe


class EventScopeProbeTests(unittest.TestCase):
    def test_splits_have_disjoint_names_templates_and_questions(self):
        parts = ("train", "cal", "test")
        rows = {part: probe.build(part) for part in parts}
        assert {part: len(rows[part]) for part in parts} == {"train": 96, "cal": 24, "test": 32}
        for index, part in enumerate(parts):
            assert {row["label"] for row in rows[part]} == set(range(4))
            for other in parts[index + 1 :]:
                assert set(probe.NAMES[part]).isdisjoint(probe.NAMES[other])
                assert {r["template"] for r in rows[part]}.isdisjoint(r["template"] for r in rows[other])
                assert {r["question"] for r in rows[part]}.isdisjoint(r["question"] for r in rows[other])


    def test_class_order_and_abstention_threshold(self):
        assert probe.LABELS == ("current", "first", "previous", "abstain")
        logits = torch.tensor([[5., 0., 0., 0.], [0., 5., 0., 0.],
                               [0., 0., 5., 0.], [0., 0., 0., 5.],
                               [0., 0., 0., 0.]])
        assert probe.choose(logits, 0.8).tolist() == [0, 1, 2, 3, 3]
        assert probe.choose(logits[:1], 1.0).tolist() == [3]
