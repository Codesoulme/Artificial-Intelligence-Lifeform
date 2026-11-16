import unittest

import torch
from scripts.deepseek_spora_writer_correction_pilot import LowRankWriter, replay_distillation
from spora.semantic_text_writer import TextSemanticWriter


class WriterCorrectionPilotTests(unittest.TestCase):
    def test_zero_initialization_and_merged_equivalence(self):
        torch.manual_seed(51)
        original = TextSemanticWriter(8)
        original_weight = original.projection.weight.detach().clone()
        writer = LowRankWriter(original, rank=2)
        inputs = torch.randn(3, 8)
        torch.testing.assert_close(writer(inputs), original(inputs))
        with torch.no_grad():
            writer.up.weight.normal_()
        torch.testing.assert_close(writer(inputs), torch.nn.functional.linear(inputs, writer.merged_weight()),
                                   atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(original.projection.weight, original_weight)
        self.assertFalse(original.projection.weight.requires_grad)

    def test_replay_distillation_only_updates_student(self):
        teacher = torch.tensor([[1., 2., 3.]], requires_grad=True)
        student = torch.tensor([[1.2, 1.8, 3.]], requires_grad=True)
        loss = replay_distillation(student, teacher)
        self.assertGreater(loss.item(), 0.)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)
        self.assertAlmostEqual(replay_distillation(teacher, teacher).item(), 0., places=6)


if __name__ == '__main__':
    unittest.main()
