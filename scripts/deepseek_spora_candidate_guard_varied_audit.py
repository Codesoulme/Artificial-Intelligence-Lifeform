"""Evaluate an isolated candidate on a newly authored, varied development set."""
from scripts import deepseek_spora_candidate_pre_read_guard_audit as candidate
from scripts import deepseek_spora_pre_read_guard_audit as audit

CASES = [
    ('你还记得小林平常爱喝哪种饮料吗？', 0),
    ('阿宁以前说过现在偏爱什么饮品？', 1),
    ('我想知道小雨喜欢喝什么。', 2),
    ('记录里面阿哲的饮料喜好是什么？', 3),
    ('你记得小林宇平时爱喝什么吗？', None),
    ('小阿宁最近喜欢什么饮料？', None),
    ('小雨婷的饮料偏好是什么？', None),
    ('关于阿哲然，他爱喝什么？', None),
    ('小林和阿宁分别喜欢什么饮料？', None),
    ('小雨与阿哲的饮品喜好一样吗？', None),
    ('有没有人喜欢喝牛奶？', None),
    ('帮我推荐一种饮料。', None),
]


def main():
    audit.CASES = CASES
    candidate.main(audit.p.ROOT / 'runs/deepseek-spora-candidate-guard-varied-audit-20261112.json')


if __name__ == '__main__':
    main()
