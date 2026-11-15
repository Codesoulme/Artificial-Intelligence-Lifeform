import unittest

from scripts import deepseek_spora_diverse_kl_correction_audit as data
from scripts import deepseek_spora_multirelation_correction_audit as multi
from scripts import deepseek_spora_router_paraphrase_audit as audit


class RouterParaphraseDesignTests(unittest.TestCase):
    def test_fresh_questions_do_not_enter_training(self):
        ordinary, _ = data.load_reviewed_prompts()
        training, fresh = audit.design(ordinary)
        self.assertEqual(len(training), 183)
        self.assertEqual(len(fresh), 23)
        self.assertFalse({row['question'] for row in training} & {row['question'] for row in fresh})
        held = [row for row in multi.examples(multi.CORRECTED)
                if row['group'] in ('old_held', 'new_held', 'unknown_relation')]
        self.assertFalse({row['question'] for row in training} & {row['question'] for row in held})


if __name__ == '__main__':
    unittest.main()
