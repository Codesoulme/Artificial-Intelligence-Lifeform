"""Train isolated temporal head with paired order and scope contrasts."""
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_temporal_gate_v2 as previous
from spora.temporal_routing import CURRENT, PAST, LATER

trainer = previous.trainer
trainer.OUTPUT = ROOT / 'runs/deepseek-spora-temporal-gate-v5-contrast-20261112'
NEW = {
    CURRENT: ('别管过去，{name}此时偏爱什么饮料？', '只问{name}如今喝什么，早些的不用说。',
              '旧饮品不查，{name}目前喜欢哪款？', '暂且不看历史，{name}当前选择的饮料是？',
              '请说现在{name}爱喝的饮品，不问过去。'),
    PAST: ('修改前{name}的饮料是什么？', '变化以前{name}喜欢哪种饮品？',
           '不问如今的，{name}更改之前偏爱什么饮料？', '后来的选择不看，请查{name}先前喝什么。',
           '在记录变动之前，{name}爱喝哪一款饮品？'),
    LATER: ('修改后{name}的饮料是什么？', '变化以后{name}喜欢哪种饮品？',
            '不问原来的，{name}更改之后偏爱什么饮料？', '先前的选择不看，请查{name}后来喝什么。',
            '在记录变动之后，{name}爱喝哪一款饮品？'),
}
trainer.EXTRA_TRAIN = {k: trainer.EXTRA_TRAIN[k] + NEW[k] for k in (CURRENT, PAST, LATER)}
trainer.TRAIN_ABSTAIN += (
    '分别说说{name}以前和如今爱喝的饮品。',
    '前后两个时间{name}都喜欢哪种饮料？',
    '别查{name}以前爱喝的饮品。',
    '把{name}变动前的饮料资料改掉。',
    '如果{name}后来爱喝果汁，请写故事。',
    '{name}饮料偏好发生变化的日期是哪天？',
    '不指定时间，讲讲{name}的饮品喜好。',
)
# Same seed and training schedule as v2 for a cleaner data-change comparison.
trainer.SEED = 510248


if __name__ == '__main__':
    with patch.object(trainer.AutoModelForCausalLM, 'from_pretrained', side_effect=previous.streamed_base), \
            patch.object(trainer, 'extract', side_effect=previous.low_memory_extract):
        trainer.run()
