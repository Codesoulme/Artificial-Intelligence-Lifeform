import unittest

import torch

from spora.research_dialogue_working_state import FIELDS, WorkingTurnState
from spora.research_observation_episode import ObservationEpisode


class ObservationEpisodeTests(unittest.TestCase):
    def test_revision_preserves_other_slots_and_serializes_no_observation_text(self):
        key = torch.tensor([.1, .7, -.3])
        started = ObservationEpisode(WorkingTurnState(), 'visit', key)
        initial = started.apply({'time': (5, 6), 'place': (7,), 'person': (8, 9)})
        edited = initial.apply({'time': (6, 10)}).apply({'time': (6, 10)})
        restored = ObservationEpisode.load(edited.payload(), topics={'visit'}, width=3, vocab=32)
        self.assertEqual(restored.revision, 2)
        self.assertEqual(restored.state.slots[FIELDS.index('time')], (6, 10))
        self.assertEqual(restored.state.slots[FIELDS.index('place')], (7,))
        self.assertEqual(restored.state.slots[FIELDS.index('person')], (8, 9))
        self.assertEqual(initial.state.slots[FIELDS.index('time')], (5, 6))
        self.assertEqual(set(edited.payload()), {'topic', 'key', 'slots', 'revision'})
        torch.testing.assert_close(restored.key, key)

    def test_restoring_state_rejects_invalid_tokens_and_vectors(self):
        episode = ObservationEpisode(WorkingTurnState().update({'time': (12,)}), 'visit', torch.ones(4))
        for payload, kwargs in (
            (episode.payload(), {'topics': {'visit'}, 'width': 4, 'vocab': 12}),
            (episode.payload(), {'topics': {'meeting'}, 'width': 4, 'vocab': 64}),
            ({**episode.payload(), 'key': torch.full((4,), torch.nan)},
             {'topics': {'visit'}, 'width': 4, 'vocab': 64}),
        ):
            with self.assertRaises(ValueError):
                ObservationEpisode.load(payload, **kwargs)


if __name__ == '__main__':
    unittest.main()
