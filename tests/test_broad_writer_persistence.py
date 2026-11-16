import unittest

from scripts.deepseek_spora_broad_writer_persistence import binding


class BroadWriterPersistenceTests(unittest.TestCase):
    def test_binding_covers_base_writer_and_each_replacement(self):
        expected = binding('base', 'writer', {'6': 'a', '12': 'b'})
        self.assertEqual(expected, binding('base', 'writer', {'12': 'b', '6': 'a'}))
        self.assertNotEqual(expected, binding('other', 'writer', {'6': 'a', '12': 'b'}))
        self.assertNotEqual(expected, binding('base', 'other', {'6': 'a', '12': 'b'}))
        self.assertNotEqual(expected, binding('base', 'writer', {'6': 'a', '12': 'c'}))


if __name__ == '__main__':
    unittest.main()
