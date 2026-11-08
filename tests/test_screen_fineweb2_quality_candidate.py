import unittest

from scripts.screen_fineweb2_quality_candidate import quality_flags


class BodyQualityTests(unittest.TestCase):
    def test_flags_site_chrome_and_commercial_stack(self):
        text = "注册登录联系客服，优惠折扣招商加盟。版权与免责声明。"
        flags = quality_flags(text)
        self.assertIn("commercial_template", flags)
        self.assertIn("site_chrome", flags)

    def test_flags_suspicious_terms_only_when_repeated(self):
        self.assertEqual(quality_flags("本文讨论彩票制度。"), [])
        self.assertIn("suspicious_terms", quality_flags("彩票博彩活动需要谨慎。"))

    def test_does_not_call_clean_prose_approved(self):
        text = "清晨的风穿过街道，树叶轻轻摇动。人们开始一天的工作。"
        self.assertEqual(quality_flags(text), [])


if __name__ == "__main__":
    unittest.main()
