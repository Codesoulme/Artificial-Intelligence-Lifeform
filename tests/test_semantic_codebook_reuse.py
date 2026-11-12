import unittest
from types import SimpleNamespace

from scripts.deepseek_spora_codebook_upper_bound_pilot import NearestTokenDecoder, pilot

torch = pilot.p.torch


class CodebookReuseTests(unittest.TestCase):
    def setUp(self):
        self.codes = torch.eye(4)
        self.decoder = NearestTokenDecoder(SimpleNamespace(codebook=self.codes), self.codes)

    def test_same_object_after_content_change_is_recomputed(self):
        query = self.codes[0].reshape(1, 1, 4).clone()
        first = self.decoder(query).clone()
        query.copy_(self.codes[1].reshape_as(query))
        second = self.decoder(query)
        self.assertEqual(int(first.argmax(-1)), 0)
        self.assertEqual(int(second.argmax(-1)), 1)

    def test_new_objects_with_different_values_do_not_reuse_previous_result(self):
        for index in (0, 2, 1, 3, 0):
            query = self.codes[index].reshape(1, 1, 4).clone()
            self.assertEqual(int(self.decoder(query).argmax(-1)), index)

    def test_zero_slot_has_no_invented_embedding(self):
        self.decoder(self.codes[0].reshape(1, 1, 4))
        empty = torch.zeros(1, 2, 4)
        self.assertTrue(torch.equal(self.decoder(empty), empty))


if __name__ == '__main__':
    unittest.main()
