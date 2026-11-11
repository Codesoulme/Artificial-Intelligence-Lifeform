"""Compare an existing observation probe before and after SPORA layer replacement.

Research only: predictions are never written to an individual.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_observation_factorized_probe import FORMAT, FactorizedProbe, combined_logits
from scripts.deepseek_observation_value_probe import ABSTAIN, WORDS, pool
from scripts.deepseek_pooled_address import extract
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.named_address import feature_files_sha256
from spora.structure_transfer import SPORALayerReplacement

PROBE = ROOT / 'runs/deepseek-observation-factorized-20261110/classifier.pt'
REPORT = PROBE.parent / 'result.json'
CHECKPOINT = ROOT / 'runs/deepseek-spora-output-v2-checkpoint-20261111/individual-with-output-v2.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-observation-transfer-audit-20261111.json'

# These examples are explicitly development challenges, not a held-out corpus.
CASES = [
    ('positive', '小林喜欢喝咖啡。', 1),
    ('positive', '阿宁最喜欢的饮料是水。', 6),
    ('positive', '小雨偏爱的饮品是酸奶。', 7),
    ('positive', '小梅最近爱喝豆浆。', 4),
    ('positive', '阿明现在最想喝的是果汁。', 3),
    ('positive', '小岚说她最喜欢喝茶。', 0),
    ('negative', '小林不喜欢喝咖啡。', ABSTAIN),
    ('negative', '小林喜欢喝咖啡吗？', ABSTAIN),
    ('negative', '要是小林喜欢喝咖啡就好了。', ABSTAIN),
    ('negative', '我不确定小林是否喜欢喝咖啡。', ABSTAIN),
    ('negative', '小林以前爱喝咖啡，现在不知道。', ABSTAIN),
    ('negative', '小林喜欢咖啡还是牛奶？', ABSTAIN),
    ('negative', '小林喜欢喝热巧克力。', ABSTAIN),
    ('negative', '小林觉得水太难喝了。', ABSTAIN),
    ('negative', '阿宁说小林不爱喝牛奶。', ABSTAIN),
    ('negative', '小林禁止我喝她的酸奶。', ABSTAIN),
    ('negative', '小林喜欢用咖啡色画画。', ABSTAIN),
    ('negative', '小林买了果汁送给阿宁。', ABSTAIN),
    ('negative', '小林说阿宁喜欢喝豆浆。', ABSTAIN),
    ('negative', '小林不再喜欢水了。', ABSTAIN),
]


def predict(base, tokenizer, probe):
    features, masks = extract(base, tokenizer, [row[1] for row in CASES], batch_size=2)
    with torch.inference_mode():
        fact, values = probe(pool(features, masks))
        classes = combined_logits(fact, values).argmax(-1)
    return [{'group': group, 'text': text, 'expected': expected,
             'predicted': int(classes[i]), 'correct': int(classes[i]) == expected,
             'fact_logit': float(fact[i])}
            for i, (group, text, expected) in enumerate(CASES)]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = torch.load(PROBE, map_location='cpu', weights_only=True)
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    if (sha(PROBE) != report['classifier_sha256'] or sha(MODEL / 'model.safetensors') != WEIGHT_SHA
            or data.get('format') != FORMAT or data.get('weight_sha256') != WEIGHT_SHA
            or data.get('feature_config_sha256') != feature_files_sha256(MODEL)
            or data.get('hidden_indices') != [6, 18] or tuple(data.get('vocabulary', ())) != WORDS):
        raise ValueError('Probe or base provenance mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    probe = FactorizedProbe(data['state']['fact.weight'].shape[1]).eval()
    probe.load_state_dict(data['state'], strict=True)
    original = predict(base, tokenizer, probe)
    decoder, organism, state = load_integrated(CHECKPOINT, base, MODEL, individual_device='cuda')
    del decoder, organism, state
    blocks = {i: layer for i, layer in enumerate(base.model.layers)
              if isinstance(layer, SPORALayerReplacement)}
    if not blocks:
        raise ValueError('Expected structural replacements')
    variant = replacement_feature_variant_sha256(MODEL, blocks)
    replaced = predict(base, tokenizer, probe)
    result = {'status': 'development_only_no_writes', 'base_sha256': WEIGHT_SHA,
              'checkpoint_sha256': sha(CHECKPOINT), 'probe_sha256': sha(PROBE),
              'replacement_layers': sorted(blocks), 'feature_variant_sha256': variant,
              'base': original, 'replaced': replaced, 'elapsed_seconds': time.perf_counter() - started,
              'limitations': ['Twenty authored examples; some match previous training templates.',
                              'A value probe alone does not determine entity or authorize a write.',
                              'No probe adaptation and no real memory writes.',
                              'Results are not evidence of general language understanding.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for name, rows in [('base', original), ('replaced', replaced)]:
        print(name, 'correct', sum(row['correct'] for row in rows), '/', len(rows),
              'negative_false_writes', sum(row['predicted'] != ABSTAIN for row in rows if row['group'] == 'negative'), flush=True)


if __name__ == '__main__':
    main()
