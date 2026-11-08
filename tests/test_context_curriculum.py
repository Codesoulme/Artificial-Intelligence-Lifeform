import unittest

from scripts.prepare_context_curriculum import OBJECTS, PLACES, rows


class ContextCurriculumTests(unittest.TestCase):
    def test_answers_follow_last_statement_and_heldout_pairs_do_not_train(self):
        held_pairs = {(o, p) for o in range(len(OBJECTS)) for p in range(len(PLACES))
                      if (o * 11 + p * 7) % 5 == 0}
        result = list(rows())
        self.assertEqual(len(result), 192)
        groups = set()
        for split, group, messages in result:
            self.assertNotIn(group, groups)
            groups.add(group)
            kind, oi, pi, *_ = group.split("-")
            oi, pi = int(oi), int(pi)
            answer_index = (pi + 3) % len(PLACES) if kind == "correction" else pi
            self.assertEqual(messages[-1]["content"], PLACES[answer_index] + "。")
            if split == "train":
                self.assertNotIn((oi, pi), held_pairs)
                self.assertNotIn((oi, answer_index), held_pairs)
            self.assertEqual([m["role"] for m in messages],
                             ["user", "assistant"] * (len(messages) // 2))


if __name__ == "__main__":
    unittest.main()
