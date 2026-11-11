"""Compare a broad-pilot layer 18 on the existing paired-layer probe."""
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_fresh_entity_graft_eval as probe
from scripts.deepseek_spora_full_stream_loader import load_full_streamed

probe.LAYER18 = ROOT / 'runs/deepseek-spora-layer18-broad-pilot-20261112/layer-18.pt'
probe.OUTPUT = ROOT / 'runs/deepseek-spora-layer18-broad-pilot-eval-20261112.json'


def streamed_base(*args, **kwargs):
    base = load_full_streamed(skip_replaced=(6, 12))
    base.to = lambda *a, **kw: base
    return base


if __name__ == '__main__':
    with patch.object(probe.AutoModelForCausalLM, 'from_pretrained', side_effect=streamed_base):
        probe.run(include_original_base=False)
