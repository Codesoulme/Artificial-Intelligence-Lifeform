import json
import subprocess
import sys
import unittest


class MultiRelationBridgeConfigTests(unittest.TestCase):
    def test_domains_have_disjoint_values_and_complete_answer_requests(self):
        # Wrappers configure a shared module; use a fresh interpreter so other
        # research test imports cannot silently change this configuration.
        code = '''
import json
from scripts.deepseek_spora_multirelation_hidden_bridge_pilot_20261117 import pilot
print(json.dumps({'train': pilot.TRAIN_VALUES, 'held': pilot.EVALUATION_VALUES,
 'groups': pilot.VALUE_GROUPS, 'words': pilot.WORDS, 'strict': pilot.STRICT_SWAP_EVAL,
 'joint': pilot.JOINT_CODEC_TRAINING, 'templates': pilot.RELATION_TEMPLATES}))
'''
        result = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
        config = json.loads(result.stdout)
        train, held = set(config['train']), set(config['held'])
        self.assertFalse(train & held)
        for values in config['groups'].values():
            self.assertEqual(len(set(values) & train), 4)
            self.assertEqual(len(set(values) & held), 2)
        self.assertEqual(train | held, set(range(len(config['words']))))
        self.assertTrue(config['strict'])
        self.assertFalse(config['joint'])
        for relation, questions in config['templates'].values():
            self.assertTrue(relation)
            self.assertEqual(len(questions), 2)
            for question in questions:
                self.assertIn('{name}', question)
                self.assertIn('完整', question)
