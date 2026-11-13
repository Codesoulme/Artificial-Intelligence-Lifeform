"""Check two original-layer name locators on unseen synthetic suffixes."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_named_binding_probe import NAMES
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_original_layer_name_router import NEAR
from scripts.deepseek_spora_original_multi_span_pilot import spans_from_mask
from spora.named_address import feature_files_sha256

BOUNDARY = ROOT / 'runs/deepseek-spora-original-name-boundary-pilot-20261114'
MULTI = ROOT / 'runs/deepseek-spora-original-multi-span-pilot-20261114'
OUTPUT = ROOT / 'runs/deepseek-spora-boundary-fresh-consensus-20261114.json'


def checked(folder, file, field):
    path = folder / file
    digest = p.sha(path)
    report = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    if digest != report[field]:
        raise ValueError('Candidate checksum mismatch')
    return torch.load(path, map_location='cpu', weights_only=True), path, digest


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    router, router_path, router_sha = checked(BOUNDARY, 'router.pt', 'router_sha256')
    multi, multi_path, multi_sha = checked(MULTI, 'locator.pt', 'locator_sha256')
    config_sha = feature_files_sha256(p.MODEL)
    if (base_sha != p.WEIGHT_SHA or router['base_sha256'] != base_sha or multi['base_sha256'] != base_sha
            or router['feature_config_sha256'] != config_sha or multi['feature_config_sha256'] != config_sha
            or router['hidden_indices'] != [6, 18] or multi['hidden_indices'] != [6, 18]):
        raise ValueError('Candidate feature provenance mismatch')
    cases = [('old_suffix', name, name[:-1]) for name in NEAR]
    cases += [('development_suffix', name + '晨', name) for name in NAMES]
    cases += [('new_suffix', name + '舟', name) for name in NAMES]
    questions = [f'你记得{name}喜欢喝哪种饮品吗？' for _, name, _ in cases]
    if len(set(questions)) != len(questions):
        raise ValueError('Duplicate questions')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                           add_generation_prompt=True) for q in questions]
    enc = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                    return_offsets_mapping=True)
    base = p.load_full_streamed().eval()
    features, masks = extract(base, tokenizer, questions, batch_size=1)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    heads = {}
    for label, state in (('boundary', router['locator_state']), ('multi', multi['state'])):
        model = nn.Linear(state['weight'].shape[1], 1).eval()
        model.load_state_dict(state, strict=True)
        heads[label] = model
    with torch.inference_mode():
        masks_by_head = {label: (model(vectors).squeeze(-1).sigmoid() >= threshold) & masks
                         for label, model, threshold in (('boundary', heads['boundary'], router['locator_threshold']),
                                                        ('multi', heads['multi'], multi['threshold']))}
    rows = []
    for i, (group, name, short_name) in enumerate(cases):
        spans = {label: spans_from_mask(mask[i], enc['offset_mapping'][i], texts[i], questions[i])
                 for label, mask in masks_by_head.items()}
        rows.append({'group': group, 'question': questions[i], 'expected_name': name,
                     'registered_prefix': short_name, 'spans': spans,
                     'boundary_exact': spans['boundary'] == [name], 'multi_exact': spans['multi'] == [name],
                     'both_short': spans['boundary'] == [short_name] and spans['multi'] == [short_name],
                     'both_equal': spans['boundary'] == spans['multi'] and len(spans['boundary']) == 1})
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(router_path) != router_sha or p.sha(multi_path) != multi_sha:
        raise ValueError('Protected input changed')
    summary = {}
    for group in sorted({r['group'] for r in rows}):
        subset = [r for r in rows if r['group'] == group]
        summary[group] = {'cases': len(subset), **{field: sum(r[field] for r in subset) for field in
                          ('boundary_exact', 'multi_exact', 'both_short', 'both_equal')}}
    result = {'summary': summary, 'rows': rows, 'base_sha256': base_sha, 'router_sha256': router_sha,
              'multi_sha256': multi_sha, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Hand-authored short names and suffixes; not a population benchmark.',
                              'Prior and development suffix cases were inspected earlier.',
                              'A matched short span is potential truncation, not a demonstrated memory read.',
                              'No projection, identity gate, intent classifier, memory, or answer tested.',
                              'Models share frozen base features and may fail together.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
