import unittest

import torch

from spora.research_temporal_intent import (FactorizedTemporalGate, past_override_mask, question_mean,
                                          temporal_character_features, temporal_logits, time_selection)


class TemporalVersionResidualTests(unittest.TestCase):
    def test_lexical_version_correction_preserves_the_read_probability(self):
        gate = FactorizedTemporalGate(4)
        with torch.no_grad():
            gate.version.weight.zero_()
            gate.version.bias.zero_()
        features = torch.randn(2, 2, 3, 2)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        texts = ('还没修改的时候填的什么？', '修改完成以后填的什么？')
        characters = temporal_character_features(texts, 64)
        weight = torch.zeros(3, 68)
        weight[1, 4:] = characters[0] - characters[1]
        original = {'architecture': 'factorized', 'classes': ['current', 'past', 'later', 'abstain'],
                    'state': gate.state_dict()}
        updated = dict(original, version_residual={'char_width': 64, 'strength': 1., 'weight': weight})
        prior = temporal_logits(original, features, mask).softmax(-1)
        corrected = temporal_logits(updated, features, mask, texts=texts).softmax(-1)
        torch.testing.assert_close(corrected[:, 3], prior[:, 3])
        self.assertGreater(corrected[0, 1], prior[0, 1])
        self.assertLess(corrected[1, 1], prior[1, 1])
        with self.assertRaisesRegex(ValueError, 'one question per feature row'):
            temporal_logits(updated, features, mask)
        updated['threshold'] = .5
        updated['version_residual']['past_override'] = .65
        guarded = temporal_logits(updated, features, mask, texts=texts).softmax(-1)
        torch.testing.assert_close(guarded, prior, rtol=0, atol=0)

    def test_characters_normalize_width_and_whitespace(self):
        features = temporal_character_features(('Ａ　甲', 'A甲'), 64)
        torch.testing.assert_close(features[0], features[1], rtol=0, atol=0)

    def test_selective_override_only_converts_admitted_nonhistorical_reads(self):
        original = torch.tensor([[.65, .10, .15, .1], [.10, .10, .70, .10],
                                 [.16, .12, .12, .60], [.15, .65, .10, .10]])
        revised = torch.tensor([[.10, .72, .08, .1], [.10, .30, .50, .10],
                                [.01, .38, .01, .60], [.10, .72, .08, .10]])
        self.assertEqual(past_override_mask(original, revised, .5, .65).tolist(), [True, False, False, False])

    def test_three_way_selector_cannot_admit_a_rejected_read(self):
        features = torch.tensor([[[[1., -1.]]], [[[-1., 1.]]]])
        mask = torch.ones((2, 1), dtype=torch.bool)
        gate = FactorizedTemporalGate(2)
        with torch.no_grad():
            gate.read.weight.copy_(torch.tensor([[2., -2.]]))
            gate.read.bias.zero_()
            gate.version.weight.zero_()
            gate.version.bias.copy_(torch.tensor([0., 0., 4.]))
        weight = torch.zeros((3, 2))
        weight[0] = 5 * question_mean(features, mask)[0]
        weight[1] = -weight[0]
        payload = {'architecture': 'factorized', 'classes': ['current', 'past', 'later', 'abstain'],
                   'threshold': .5, 'state': gate.state_dict()}
        selector = {'char_width': 0, 'threshold': .55, 'weight': weight}
        original = temporal_logits(payload, features, mask).softmax(-1)
        revised = temporal_logits(dict(payload, version_selector=selector), features, mask).softmax(-1)
        self.assertEqual(time_selection(original, .5, factorized=True).tolist(), [2, 3])
        self.assertEqual(time_selection(revised, .5, factorized=True).tolist(), [0, 3])
        torch.testing.assert_close(revised[:, 3], original[:, 3])
