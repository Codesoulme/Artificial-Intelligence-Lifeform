import unittest
from scripts.deepseek_spora_interleaved_routed_audit import make_cases, score


class InterleavedRouteAuditTests(unittest.TestCase):
    def test_expected_routes_cover_each_entity_and_phase(self):
        cases = make_cases()
        positives = [case for case in cases if case['index'] is not None]
        self.assertEqual(len(positives), 16)
        self.assertEqual({(case['index'], case['phase']) for case in positives},
                         {(index, phase) for index in range(8) for phase in (0, 1)})
        self.assertEqual(len(cases) - len(positives), 6)

    def test_answer_cannot_be_correct_when_address_or_time_is_wrong(self):
        case = {'index': 0, 'phase': 0, 'mode': 1}
        events = {(0, 0): {'key': 6, 'time': 0, 'value': '茶'}}
        good = {'accepted': True, 'key': 6, 'temporal_intent': 1,
                'memory_event_selection': {'event_time': 0}, 'answer': '茶。'}
        self.assertTrue(score(case, good, events)['answer_exact'])
        self.assertFalse(score(case, {**good, 'key': 9}, events)['answer_exact'])
        self.assertFalse(score(case, {**good, 'memory_event_selection': {'event_time': 2}}, events)['answer_exact'])
        self.assertFalse(score(case, {**good, 'answer': '茶杯'}, events)['answer_exact'])


if __name__ == '__main__':
    unittest.main()
