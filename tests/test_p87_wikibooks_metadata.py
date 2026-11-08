import bz2
import tempfile
import unittest
from pathlib import Path

from scripts.p87_wikibooks_metadata import classify, group_hash, read_metadata


class MetadataInventoryTests(unittest.TestCase):
    def test_routing_is_fixed_and_ambiguity_explicit(self):
        self.assertEqual(classify('数学与物理')['category'], 'C')
        self.assertEqual(classify('数学与物理')['category_ambiguity'], 'multiple_categories')
        self.assertEqual(classify('某书')['category'], 'F')
        self.assertEqual(classify('某书')['category_ambiguity'], 'no_title_keyword')

    def test_whole_prefix_excluded_and_bucket_and_redirect_skipped(self):
        reserved = next('保留' + str(i) for i in range(1000)
                        if int(group_hash('保留' + str(i))[:8], 16) % 10 == 0)
        allowed = next('可用' + str(i) for i in range(1000)
                       if int(group_hash('可用' + str(i))[:8], 16) % 10 != 0)
        old = next('旧样本' + str(i) for i in range(1000)
                   if int(group_hash('旧样本' + str(i))[:8], 16) % 10 != 0)
        def page(title, extra=''):
            return f'<page><title>{title}</title><ns>0</ns>{extra}<revision><id>42</id><text>SECRET BODY</text></revision></page>'
        xml = ('<mediawiki>' + page(allowed + '/一') + page(allowed + '/二') +
               page(old + '/一') + page(old + '/二') + page(reserved + '/一') +
               page('转向', '<redirect title="X"/>') + '</mediawiki>')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'sample.bz2'
            path.write_bytes(bz2.compress(xml.encode('utf-8')))
            counts, groups = read_metadata(path, {old, group_hash(old)})
        self.assertEqual(counts['eligible_metadata_pages'], 2)
        self.assertEqual(counts['old_sample_prefix_pages'], 2)
        self.assertEqual(counts['reserved_bucket_pages'], 1)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['page_count'], 2)
        self.assertNotIn('SECRET BODY', repr(groups))
        self.assertEqual(groups[0]['pages'][0]['revision_id'], '42')


if __name__ == '__main__':
    unittest.main()
