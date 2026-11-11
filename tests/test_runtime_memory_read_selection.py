import unittest
from unittest.mock import patch

import torch

from scripts import deepseek_spora_runtime_chat as runtime
from tests.test_deepseek_memory_page_selection import FakeOrganism


class RuntimeMemoryReadSelectionTests(unittest.TestCase):
    def setUp(self):
        self.organism = FakeOrganism()
        self.state = type('State', (), {'cursor': 130, 'pages': [None, None, None]})()

    def test_current_and_history_share_the_selected_key_and_event(self):
        evidence = torch.tensor([[0.2, 0.8]])
        for mode, event, expected_mode, expected_time, expected_slot in (
                ('current', None, 0, 0, 0), ('past', 64, 1, 64, 2), ('later', 129, 1, 129, 3)):
            with self.subTest(mode=mode), patch.object(runtime, 'query', return_value=evidence) as query:
                args, actual = runtime._prepare_memory_read(self.organism, self.state, 1, mode, event)
                self.assertIs(args['memory_state'], self.state)
                self.assertEqual(args['memory_slots'].tolist(), [expected_slot])
                self.assertIs(actual, evidence)
                organism, state, keys, modes, times = query.call_args.args
                self.assertIs(organism, self.organism)
                self.assertIs(state, self.state)
                self.assertEqual((keys.tolist(), modes.tolist(), times.tolist()),
                                 ([1], [expected_mode], [expected_time]))

    def test_invalid_selection_does_not_query(self):
        for mode, event in (('unknown', 0), ('current', 0), ('past', None), ('later', -1),
                            ('past', True), ('past', 130)):
            with self.subTest(mode=mode, event=event), patch.object(runtime, 'query') as query:
                with self.assertRaises(ValueError):
                    runtime._prepare_memory_read(self.organism, self.state, 1, mode, event)
                query.assert_not_called()


if __name__ == '__main__':
    unittest.main()
