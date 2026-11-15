import collections
import unittest

from scripts import deepseek_spora_multirelation_correction_audit as audit


class MultirelationDesignTests(unittest.TestCase):
    def test_disjoint_known_and_unknown_and_changed_values(self):
        audit.validate_design()
        changed = {key for key in audit.LEARNED if audit.LEARNED[key] != audit.CORRECTED[key]}
        self.assertEqual(changed, {('小林', 'city'), ('小乔', 'hobby')})
        for facts in (audit.LEARNED, audit.CORRECTED):
            rows = audit.examples(facts)
            self.assertEqual(len({row['question'] for row in rows}), len(rows))
            counts = collections.Counter(row['group'] for row in rows)
            self.assertEqual(counts['old_train'], 6)
            self.assertEqual(counts['old_held'], 6)
            self.assertEqual(counts['new_train'], 8)
            self.assertEqual(counts['new_held'], 8)
            self.assertEqual(counts['unknown_person'], 12)
            self.assertEqual(counts['unknown_relation'], 8)

    def test_fixed_replay_schedule_covers_every_known_pair_and_template(self):
        counts = collections.Counter(((step // 2) % len(audit.LEARNED), step % 2)
                                     for step in range(audit.STEPS))
        self.assertEqual(len(counts), 14)
        self.assertEqual(set(counts.values()), {6})


if __name__ == '__main__':
    unittest.main()
