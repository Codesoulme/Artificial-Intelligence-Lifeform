import unittest

from scripts import deepseek_spora_name_extraction_audit as audit
from spora.identity_fingerprints import IdentityFingerprints


class NameExtractionAuditTests(unittest.TestCase):
    def test_parser_fails_closed_on_non_json_and_invalid_names(self):
        question = '小林月喜欢什么？'
        self.assertEqual(audit.parse_response('{"name":"小林月"}', question), ('小林月', 'name'))
        self.assertEqual(audit.parse_response('{"name":null}', question), (None, 'null'))
        self.assertEqual(audit.parse_response('小林月', question), (None, 'invalid_json'))
        self.assertEqual(audit.parse_response('{"name":"阿宁"}', question), (None, 'invalid_name'))
        self.assertEqual(audit.parse_response('{"name":"小林","name":"小林月"}', question),
                         (None, 'invalid_json'))

    def test_substring_acceptance_is_not_full_name_verification(self):
        name, status = audit.parse_response('{"name":"小林"}', '小林月喜欢什么？')
        self.assertEqual(status, 'name')
        fingerprints = IdentityFingerprints(1)
        fingerprints.bind(0, '小林')
        self.assertEqual(fingerprints.lookup(name), 0)
        self.assertIsNone(fingerprints.lookup('小林月'))

    def test_embedded_json_requires_one_object(self):
        question = '小林月喜欢什么？'
        self.assertEqual(audit.parse_embedded_response('结果：```json\n{"name":"小林月"}\n```', question),
                         ('小林月', 'name'))
        self.assertEqual(audit.parse_embedded_response('{"name":"小林"} {"name":"小林月"}', question),
                         (None, 'ambiguous_json'))
        self.assertEqual(audit.parse_embedded_response('说明 {"name":"小林","name":"小林月"}', question),
                         (None, 'invalid_json'))


if __name__ == '__main__':
    unittest.main()
