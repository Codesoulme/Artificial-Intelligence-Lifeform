import unittest
from scripts.screen_fineweb2_sample import reasons


class FinewebScreenTests(unittest.TestCase):
    def test_rejects_sensitive_and_bad_text(self):
        row = {"text": "甲" * 120 + " test@example.com 13800138000"}
        self.assertIn("email_pattern", reasons(row)); self.assertIn("phone_pattern", reasons(row))
    def test_accepts_basic_clean_text(self):
        text = "。".join(f"这是第{i}段中文说明，内容用于测试正常的连续文本筛选。" for i in range(20))
        self.assertEqual(reasons({"text": text}), [])


if __name__ == "__main__": unittest.main()
