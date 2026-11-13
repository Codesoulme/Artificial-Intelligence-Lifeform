import unittest

import torch

from scripts.deepseek_spora_semantic_bridge_pilot import aligned_teacher_kl, explicit_teacher_question


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


if __name__ == '__main__':
    unittest.main()
