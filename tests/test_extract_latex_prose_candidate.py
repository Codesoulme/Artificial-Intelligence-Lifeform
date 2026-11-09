import unittest

from scripts.extract_latex_prose_candidate import FORMULA, clean_latex, extract_paragraphs


class LatexCandidateTests(unittest.TestCase):
    def test_clean_latex_preserves_text_and_structural_labels(self):
        raw = r'''\section{标题}
这是一个足够长的中文段落，用来说明对象和证明过程。它包含一个公式 $x^2+y^2$，公式应该被明确占位，而不是悄悄删除。这里继续补充文字以满足候选阈值。

\begin{proof}
证明：这是证明中的中文说明，也要保留在结构化结果里，并且长度足够让测试稳定通过。
\end{proof}'''
        cleaned = clean_latex(raw)
        self.assertIn('标题：标题', cleaned)
        self.assertIn(FORMULA, cleaned)
        self.assertNotIn('\\section', cleaned)
        self.assertNotIn('compactitem', cleaned)
        self.assertIn('证明：', cleaned)

    def test_extract_rejects_short_or_low_cjk_blocks(self):
        rows = extract_paragraphs('short\n\n' + ('中文内容' * 30))
        self.assertEqual(len(rows), 1)
        self.assertGreaterEqual(rows[0]['cjk'], 40)


if __name__ == '__main__':
    unittest.main()
