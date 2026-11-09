"""Run the structural SPORA memory probe against the DeepSeek baseline."""
from scripts import open_base_memory_probe as probe
from scripts.deepseek_local_baseline import MODEL, REVISION, ROOT, WEIGHT_SHA

probe.MODEL = MODEL
probe.MODEL_NAME = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
probe.REVISION = REVISION
probe.WEIGHT_SHA = WEIGHT_SHA
probe.ATTENTION = 'sdpa'
probe.OUTPUT = ROOT / 'runs/deepseek-r1-1.5b-memory-probe-20261109.json'

if __name__ == '__main__':
    probe.run()
