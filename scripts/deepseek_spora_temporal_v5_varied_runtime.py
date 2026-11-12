"""Compare the v5 temporal candidate on newly authored runtime questions."""
from scripts import deepseek_spora_temporal_regression_runtime as audit


def main():
    audit.CANDIDATE_DIR = audit.ROOT / 'runs/deepseek-spora-temporal-gate-v5-contrast-20261112'
    audit.OUTPUT = audit.ROOT / 'runs/deepseek-spora-temporal-v5-varied-runtime-20261112.json'
    audit.QUESTIONS = (
        ('current', '目前{name}最喜欢哪种饮料？', 0, '水'),
        ('past', '更换喜好之前，{name}喜欢喝什么饮料？', 1, '咖啡'),
        ('later', '最后一次更新后，{name}喜欢什么饮品？', 2, '水'),
    )
    audit.main()


if __name__ == '__main__':
    main()
