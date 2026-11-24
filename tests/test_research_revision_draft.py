import unittest

from spora.research_revision_draft import RevisionDecision as Decision, RevisionDraft


class RevisionDraftTests(unittest.TestCase):
    def test_confirmed_then_withdrawn_has_no_patch(self):
        draft = RevisionDraft('place').observe(Decision.CONFIRM, field='place', value='北门')
        self.assertEqual(draft.patch(), {'place': '北门'})
        withdrawn = draft.observe(Decision.WITHDRAW)
        self.assertEqual(withdrawn.patch(), {})
        self.assertEqual(withdrawn.observe(Decision.CONFIRM).patch(), {})
        self.assertEqual(draft.patch(), {'place': '北门'})

    def test_proposal_only_applies_after_confirmation(self):
        draft = RevisionDraft('place').observe(Decision.PROPOSE, field='place', value='北门')
        self.assertEqual(draft.patch(), {})
        self.assertEqual(draft.observe(Decision.CONFIRM).patch(), {'place': '北门'})

    def test_unrelated_turn_and_different_field_preserve_prior_decision(self):
        draft = RevisionDraft('place').observe(Decision.CONFIRM, field='place', value='北门')
        self.assertIs(draft.observe(Decision.OTHER), draft)
        self.assertIs(draft.observe(Decision.WITHDRAW, field='time'), draft)
        self.assertEqual(draft.observe(Decision.PROPOSE).patch(), {})
        self.assertEqual(draft.observe(Decision.PROPOSE).observe(Decision.CONFIRM).patch(), {'place': '北门'})


if __name__ == '__main__':
    unittest.main()
