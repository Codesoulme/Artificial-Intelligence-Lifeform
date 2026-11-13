import unittest

from spora.explicit_span_address import explicit_span_address
from spora.identity_fingerprints import IdentityFingerprints


class ExplicitSpanAddressTests(unittest.TestCase):
    def setUp(self):
        self.names = IdentityFingerprints(4)
        self.names.bind(1, '小林')
        self.names.bind(2, '小林舟')
        self.question = '小林舟喜欢什么？'

    def test_complete_explicit_selection_and_missing(self):
        self.assertEqual(explicit_span_address(self.names, self.question, (0, 3), '小林舟')['key'], 2)
        self.assertEqual(explicit_span_address(self.names, self.question, None, None)['reason'],
                         'explicit_selection_required')

    def test_rejects_invalid_bounds_and_unregistered_name(self):
        for span, name in (((0, 2), '小林舟'), ((-1, 2), '小林'), ((True, 2), '小林'),
                           ((0, 30), '小林'), ((0, 2), None)):
            self.assertEqual(explicit_span_address(self.names, self.question, span, name)['reason'],
                             'invalid_selection')
        self.assertEqual(explicit_span_address(self.names, '小雨喜欢茶', (0, 2), '小雨')['reason'],
                         'explicit_name_not_enrolled')

    def test_wrong_user_selection_can_still_resolve_short_name(self):
        self.assertEqual(explicit_span_address(self.names, self.question, (0, 2), '小林')['key'], 1)


if __name__ == '__main__':
    unittest.main()
