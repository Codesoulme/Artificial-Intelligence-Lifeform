from datetime import date
import unittest

import torch

from spora.research_version_roles import (ABSTAIN, CURRENT, DENY, EXCLUDE, EXPIRED, FIRST, PREVIOUS, SEEK,
                                          UNSPECIFIED, VersionRoleHead, apply_version_plan, clause_spans, version_plan)


class VersionRolesTests(unittest.TestCase):
    def test_role_evidence_does_not_replace_version_evidence(self):
        torch.manual_seed(2)
        head = VersionRoleHead(4)
        mean = torch.tensor([[[1., 2., 5., 0.]]])
        tail = mean.flip(-1)
        original_roles, original_versions = head(mean)
        moved_roles, moved_versions = head(mean, tail)
        self.assertFalse(torch.equal(original_roles, moved_roles))
        self.assertTrue(torch.equal(original_versions, moved_versions))

    def test_role_switch_changes_target_without_changing_the_version_mentions(self):
        self.assertEqual(version_plan([SEEK, EXCLUDE], [FIRST, CURRENT]), FIRST)
        self.assertEqual(version_plan([EXCLUDE, SEEK], [FIRST, CURRENT]), CURRENT)
        self.assertEqual(version_plan([SEEK, EXCLUDE], [PREVIOUS, EXPIRED]), ABSTAIN)
        self.assertEqual(version_plan([SEEK, EXCLUDE], [CURRENT, CURRENT]), ABSTAIN)

    def test_generic_lookup_defaults_only_when_all_expired_versions_are_excluded(self):
        self.assertEqual(version_plan([EXCLUDE, SEEK], [EXPIRED, UNSPECIFIED]), CURRENT)
        self.assertEqual(version_plan([EXCLUDE, SEEK], [CURRENT, UNSPECIFIED]), ABSTAIN)
        self.assertEqual(version_plan([DENY, SEEK], [UNSPECIFIED, CURRENT]), ABSTAIN)

    def test_learned_plan_cannot_override_an_explicit_date_or_nonread_request(self):
        today = date(2026, 10, 26)
        revisions = ((date(2026, 10, 2), date(2026, 10, 2)), (date(2026, 10, 7), date(2026, 10, 7)))
        self.assertEqual(apply_version_plan('查霍盈的饮品，2026-10-01有效的是什么？', revisions,
                                            today, CURRENT, CURRENT), (None, 'no_matching_version'))
        self.assertEqual(apply_version_plan('请勿读取霍盈的饮品，告诉我怎么画表。', revisions,
                                            today, CURRENT, FIRST), (None, 'outside_lookup_scope'))
        self.assertEqual(apply_version_plan('先排除现行。霍盈的饮品记录是什么？', revisions,
                                            today, CURRENT, FIRST), (0, 'role_plan'))
        question = '排除旧版。霍盈的饮品记录是什么？'
        self.assertEqual([question[a:b] for a, b in clause_spans(question)], ['排除旧版', '霍盈的饮品记录是什么'])
