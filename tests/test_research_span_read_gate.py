import unittest

import torch

from spora.research_dialogue_working_state import FIELDS
from spora.research_span_read_gate import StateReadGate


class StateReadGateTests(unittest.TestCase):
    def test_same_question_rejects_wrong_topic_and_missing_field(self):
        field_scores = torch.full((5, len(FIELDS)), -2.)
        present = torch.zeros_like(field_scores, dtype=torch.bool)
        required = [FIELDS.index('time'), FIELDS.index('place')]
        field_scores[:, required] = 2.
        present[:, required] = True
        present[2, required[0]] = False
        present[3] = False
        need = torch.tensor([1., 1., 1., 1., -1.])
        relevance = torch.tensor([1., -1., 1., 1., 1.])
        permitted = StateReadGate.permits(need, field_scores, relevance, present, (0., 0., 0.))
        self.assertEqual(permitted.tolist(), [True, False, False, False, False])

    def test_gate_abstains_when_no_field_is_predicted(self):
        need = match = torch.tensor(5.)
        empty_requirements = torch.full((len(FIELDS),), -1.)
        all_present = torch.ones(len(FIELDS), dtype=torch.bool)
        self.assertFalse(bool(StateReadGate.permits(need, empty_requirements, match, all_present, (0., 0., 0.))))


if __name__ == '__main__':
    unittest.main()
