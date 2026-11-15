import unittest

import torch

from scripts.deepseek_spora_semantic_bridge_pilot import (
    aligned_teacher_hidden_delta,
    aligned_teacher_hidden_cosine,
    aligned_teacher_kl,
    explicit_teacher_question,
    remap_condition_weight,
    select_condition_slots,
)
from scripts.deepseek_spora_joint_codec_bridge_pilot import JointCodec


class SemanticTeacherAlignmentTests(unittest.TestCase):
    def test_teacher_only_prompt_has_fact_and_preserves_question(self):
        question = '小林喜欢的水果是什么？'
        result = explicit_teacher_question(question, '小林', '山竹')
        self.assertIn('小林喜欢的水果是山竹。', result)
        self.assertTrue(result.endswith(question))
        self.assertNotIn('山竹', question)

    def test_kl_is_position_aligned_and_does_not_train_teacher(self):
        teacher = torch.tensor([[[2., -1.], [-1., 2.]]], requires_grad=True)
        student = torch.tensor([[[0., 0.], [0., 0.]]], requires_grad=True)
        loss = aligned_teacher_kl(student, teacher, 2.0)
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(student.grad.norm()), 0.0)
        self.assertLess(float(aligned_teacher_kl(teacher.detach(), teacher, 2.0).abs()), 1e-5)
        with self.assertRaises(ValueError):
            aligned_teacher_kl(student, teacher[:, :1], 2.0)

    def test_hidden_alignment_trains_student_only(self):
        teacher = torch.tensor([[[2., 0.], [0., 2.]]], requires_grad=True)
        student = torch.tensor([[[0., 1.], [1., 1.]]], requires_grad=True)
        loss = aligned_teacher_hidden_cosine(student, teacher)
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(student.grad.norm()), 0.0)
        self.assertLess(float(aligned_teacher_hidden_cosine(teacher.detach(), teacher).abs()), 1e-5)
        with self.assertRaises(ValueError):
            aligned_teacher_hidden_cosine(student, teacher[:, :1])

    def test_hidden_delta_alignment_uses_a_detached_baseline(self):
        baseline = torch.tensor([[[1., 1.], [1., 1.]]])
        teacher = torch.tensor([[[2., 1.], [1., 3.]]])
        student = (baseline + torch.tensor([[[0., 1.], [1., 0.]]])).requires_grad_()
        loss = aligned_teacher_hidden_delta(student, baseline, teacher)
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertGreater(float(student.grad.norm()), 0.0)
        with self.assertRaises(ValueError):
            aligned_teacher_hidden_delta(student, baseline[:, :1], teacher)

    def test_condition_lane_can_exclude_identity_slot(self):
        memory = torch.arange(30., dtype=torch.float32).reshape(1, 5, 6)
        selected = select_condition_slots(memory, 5, 3, 1)
        expected = torch.nn.functional.normalize(memory[:, 1:4], dim=-1)
        self.assertTrue(torch.equal(selected, expected))
        with self.assertRaises(ValueError):
            select_condition_slots(memory, 5, 4, 2)

    def test_condition_weight_remap_moves_fact_columns_and_zeros_tail(self):
        import scripts.deepseek_spora_semantic_bridge_pilot as pilot
        old_flag = pilot.REMAP_CONDITION_INIT
        pilot.REMAP_CONDITION_INIT = True
        try:
            old = torch.arange(8., dtype=torch.float32).reshape(2, 4)
            target = torch.empty(2, 4)
            mapped = remap_condition_weight(old, target, 1, 1)
            self.assertTrue(torch.equal(mapped, torch.tensor([[1., 2., 3., 0.], [5., 6., 7., 0.]])))
        finally:
            pilot.REMAP_CONDITION_INIT = old_flag

    def test_joint_codec_projection_receives_gradient(self):
        codec = JointCodec(torch.randn(10, 8), memory_width=4, slot_count=2, eos_token_id=9)
        encoded = codec.encode_differentiable(torch.tensor([1]))
        encoded.square().sum().backward()
        self.assertIsNotNone(codec.projection.grad)
        self.assertGreater(float(codec.projection.grad.abs().sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
