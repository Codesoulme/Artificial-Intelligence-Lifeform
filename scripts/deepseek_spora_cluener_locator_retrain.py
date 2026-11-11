"""Retrain the name locator on representations after SPORA replacement."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch

from scripts import deepseek_cluener_locator_train as trainer
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.structure_transfer import SPORALayerReplacement

CHECKPOINT = ROOT / 'runs/deepseek-spora-output-v2-checkpoint-20261111/individual-with-output-v2.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-cluener-locator-retrained-20261111'

_original_extract = trainer.extract
_loaded = False
_variant = None


def extract_replaced(base, tokenizer, questions, batch_size=8, **kwargs):
    global _loaded, _variant
    if not _loaded:
        decoder, organism, state = load_integrated(CHECKPOINT, base, MODEL, individual_device='cuda')
        del decoder, organism, state
        replacements = {i: layer for i, layer in enumerate(base.model.layers)
                        if isinstance(layer, SPORALayerReplacement)}
        _variant = replacement_feature_variant_sha256(MODEL, replacements)
        _loaded = True
    return _original_extract(base, tokenizer, questions, batch_size=batch_size, **kwargs)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA or sha(CHECKPOINT) == '':
        raise ValueError('Protected base or checkpoint unavailable')
    trainer.OUTPUT = OUTPUT
    trainer.REQUIRE_BASELINE_REPRODUCTION = False
    trainer.extract = extract_replaced
    trainer.run()
    locator_path = OUTPUT / 'locator.pt'
    data = torch.load(locator_path, map_location='cpu', weights_only=True)
    data['format'] = 'spora-name-span-locator-replaced-feature-development-v1'
    data['source_checkpoint_sha256'] = sha(CHECKPOINT)
    data['feature_variant_sha256'] = _variant
    torch.save(data, locator_path)
    report_path = OUTPUT / 'result.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    report['format'] = data['format']
    report['source_checkpoint_sha256'] = data['source_checkpoint_sha256']
    report['feature_variant_sha256'] = _variant
    report['locator_sha256'] = sha(locator_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'locator_sha256': report['locator_sha256'], 'feature_variant_sha256': _variant}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
