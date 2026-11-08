import unittest
from scripts.partition_clean_pretrain_candidate import classify,INPUT_SHA
class PartitionTests(unittest.TestCase):
 def test_categories_are_conservative(self):
  self.assertEqual(classify('请写一个Python函数。```python\nprint(1)\n```')[0],'code_or_markup')
  self.assertEqual(classify('春风吹过安静的山谷，树叶在阳光下轻轻摇动。远处的溪水绕过石桥，村民沿着小路慢慢回家。夜色降临时，山谷重新恢复宁静。山坡上的野花在暮色里逐渐看不清，溪边留下了白天行人的脚印。远处的灯光一盏盏亮起，屋檐下传来收拾晚餐的声音，空气里有淡淡的柴火气息。')[0],'prose_candidate')
  self.assertEqual(len(INPUT_SHA),64)
if __name__=='__main__':unittest.main()
