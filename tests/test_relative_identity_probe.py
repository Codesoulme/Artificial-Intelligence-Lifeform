import unittest

from scripts.deepseek_relative_identity_probe import ADDRESSEE, SPEAKER, UNKNOWN, resolve_relative


class RelativeIdentityTests(unittest.TestCase):
    def test_role_swap_preserves_self_and_explicit_participant(self):
        self.assertEqual(resolve_relative(SPEAKER, 'user', 6, 22, 64), 22)
        self.assertEqual(resolve_relative(ADDRESSEE, 'user', 6, 22, 64), 6)
        self.assertEqual(resolve_relative(SPEAKER, 'assistant', 6, 22, 64), 6)
        self.assertEqual(resolve_relative(ADDRESSEE, 'assistant', 6, 22, 64), 22)
        self.assertEqual(resolve_relative(ADDRESSEE, 'user', 6, 35, 64), 6)
        self.assertEqual(resolve_relative(SPEAKER, 'user', 6, 35, 64), 35)
        self.assertIsNone(resolve_relative(UNKNOWN, 'user', 6, 22, 64))

    def test_missing_or_colliding_identity_is_rejected(self):
        for role, own, participant in (('user', 6, 6), ('user', 6, None), ('guest', 6, 22), ('user', 6, 64)):
            with self.assertRaises(ValueError):
                resolve_relative(SPEAKER, role, own, participant, 64)


if __name__ == '__main__':
    unittest.main()
