"""Calibrate a candidate name router on separate authored development phrases."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_migrate import OUTPUT as NAMED
from scripts.deepseek_named_binding_probe import KEYS, NAMES, UNKNOWN, pooled, predict
from scripts.deepseek_named_incremental_probe import NEW_KEYS
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_memory_answer_distill import _load_candidate
from scripts.deepseek_spora_recalibrate_name_router import SOURCE, OUTPUT as ROUTER
from spora.growing import load
from spora.integrated_checkpoint import replacement_feature_variant_sha256
from spora.language_training import atomic_save
from spora.named_address import address_basis_sha256, feature_files_sha256

OUTPUT = ROOT/'runs/deepseek-spora-name-threshold-calibration-20261111'
CAL_TEMPLATES = ('{name}最近记下的饮料偏好是什么？', '旧记录里的{name}钟意哪种饮料？')
CAL_NO_NAME = ('把我刚才的笔记概括一下。', '现在记录的饮料是什么？', '聊聊你最近学了什么。', '请解释什么是历史记忆。')
CAL_MULTIPLE = ('小林和阿宁分别偏爱什么饮品？', '小雨与阿哲的饮料有什么区别？',
                '小梅和阿远都喝什么？', '小岚与阿明以前各喝什么？',
                '阿松和小雪现在的饮料记录？', '小青和阿禾更喜欢哪些饮品？')


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model mismatch')
    info = json.loads((ROUTER/'result.json').read_text(encoding='utf-8'))
    if sha(ROUTER/'router.pt') != info['router_sha256']:
        raise ValueError('Router checkpoint mismatch')
    payload = torch.load(ROUTER/'router.pt', map_location='cpu', weights_only=True)
    config_hash = feature_files_sha256(MODEL)
    organism, _, _ = load(NAMED/'individual.pt', 'cpu')
    if (payload.get('base_sha256') != WEIGHT_SHA or payload.get('feature_config_sha256') != config_hash
            or payload.get('basis_sha256') != address_basis_sha256(organism)):
        raise ValueError('Router provenance mismatch')
    bases = torch.stack([F.normalize(site.current_basis.detach().float(), dim=-1) for site in organism.memories])
    calibration = []
    for name, key in zip(NAMES+UNKNOWN[:2], KEYS+NEW_KEYS):
        calibration += [{'question': t.format(name=name), 'expected_key': key, 'group': 'known'} for t in CAL_TEMPLATES]
    for name in UNKNOWN[2:]:
        calibration += [{'question': t.format(name=name), 'expected_key': None, 'group': 'unbound'} for t in CAL_TEMPLATES]
    calibration += [{'question': q, 'expected_key': None, 'group': 'no_name'} for q in CAL_NO_NAME]
    calibration += [{'question': q, 'expected_key': None, 'group': 'multiple_names'} for q in CAL_MULTIPLE]
    held = [{'question': r['question'], 'expected_key': r['expected_key'], 'group': r['group']}
            for r in info['results']['recalibrated_router']['rows']]
    if {row['question'] for row in calibration} & {row['question'] for row in held}:
        raise ValueError('Calibration and held text overlap')
    rows = calibration+held
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    blocks = {i: _load_candidate(SOURCE/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    if replacement_feature_variant_sha256(MODEL, blocks) != payload['feature_variant_sha256']:
        raise ValueError('Router and structural variant mismatch')
    for i, block in blocks.items():
        base.model.layers[i] = block
    features, masks = extract(base, tokenizer, [row['question'] for row in rows], batch_size=4)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    locator = nn.Linear(vectors.shape[-1], 1).eval()
    locator.load_state_dict(payload['locator_state'], strict=True)
    scored = []
    with torch.inference_mode():
        scores = locator(vectors).squeeze(-1).sigmoid()
        for index, row in enumerate(rows):
            selected, single = single_span_mask(scores[index], masks[index], payload['locator_threshold'])
            result = {**row, 'single_span': single, 'indices': None, 'confidence': None}
            if single:
                span = pooled(features[index:index+1], selected.unsqueeze(0))
                indices, similarity = predict(span, payload['projection'], bases)
                result['indices'], result['confidence'] = indices[0].tolist(), float(similarity[0].min())
            scored.append(result)
    calibration_rows, held_rows = scored[:len(calibration)], scored[len(calibration):]
    negatives = [row['confidence'] for row in calibration_rows if row['expected_key'] is None
                 and row['confidence'] is not None and row['indices'][0] == row['indices'][1]]
    floor = float(info['results']['prior_router']['threshold'])
    threshold = max(floor, (max(negatives)+1e-6) if negatives else floor)
    if not 0 <= threshold <= 1:
        raise ValueError('Calibration cannot produce a usable threshold')

    def summarize(items, cutoff):
        groups = {}
        for row in items:
            row = dict(row)
            row['accepted'] = bool(row['single_span'] and row['indices'][0] == row['indices'][1]
                                   and row['confidence'] >= cutoff)
            row['correct'] = row['expected_key'] is not None and row['accepted'] and row['indices'] == [row['expected_key']]*2
            bucket = groups.setdefault(row['group'], {'total': 0, 'accepted': 0, 'correct': 0, 'rows': []})
            bucket['total'] += 1
            bucket['accepted'] += int(row['accepted'])
            bucket['correct'] += int(row['correct'])
            bucket['rows'].append(row)
        return groups

    results = {'training_threshold': {'threshold': payload['threshold'],
                                      'calibration': summarize(calibration_rows, payload['threshold']),
                                      'held': summarize(held_rows, payload['threshold'])},
               'calibrated_threshold': {'threshold': threshold, 'calibration': summarize(calibration_rows, threshold),
                                        'held': summarize(held_rows, threshold)}}
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT/'threshold.pt'
    atomic_save({'format': 'spora-name-threshold-calibration-v1', 'router_sha256': sha(ROUTER/'router.pt'),
                 'feature_variant_sha256': payload['feature_variant_sha256'], 'threshold': threshold,
                 'source_floor': floor, 'negative_eligible': len(negatives)}, checkpoint)
    report = {'results': results, 'router_sha256': sha(ROUTER/'router.pt'), 'threshold_sha256': sha(checkpoint),
              'feature_variant_sha256': payload['feature_variant_sha256'], 'base_sha256': WEIGHT_SHA,
              'calibration_negative_eligible': len(negatives), 'max_negative_score': max(negatives) if negatives else None,
              'elapsed_seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Small, manually authored calibration and previously inspected development tests.',
                              'Threshold floor uses earlier router threshold; calibration negatives are not exhaustive.',
                              'Names and addresses are supervised; no natural-language write or temporal routing tested.',
                              'Similarity threshold is not a calibrated probability or an access control boundary.']}
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'threshold': threshold, 'negative_eligible': len(negatives),
                      'held': {g: {k: v for k, v in x.items() if k != 'rows'}
                               for g, x in results['calibrated_threshold']['held'].items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
