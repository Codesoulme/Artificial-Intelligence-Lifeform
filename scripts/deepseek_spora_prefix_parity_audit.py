"""Compare reduced-memory prefix outputs against a saved full-runtime audit."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_spora_prefix_features import load_prefix

REFERENCE = ROOT / 'runs/deepseek-spora-default-near-name-audit-20261111.json'
SPANS = ROOT / 'runs/deepseek-spora-default-near-name-spans-20261111.json'
OUTPUT = ROOT / 'runs/deepseek-spora-prefix-parity-audit-20261111.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    ref = json.loads(REFERENCE.read_text(encoding='utf-8'))
    spans = json.loads(SPANS.read_text(encoding='utf-8'))
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    if ref['source_checkpoint_sha256'] != checkpoint_sha or spans['source_checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Reference checkpoint mismatch')
    entity_file = runtime.ENTITY / 'gate.pt'
    entity_report = json.loads((runtime.ENTITY / 'result.json').read_text(encoding='utf-8'))
    if runtime.sha(entity_file) != entity_report['gate_sha256']:
        raise ValueError('Entity gate digest mismatch')
    entity_data = torch.load(entity_file, map_location='cpu', weights_only=True)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(runtime.MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model, router, variant = load_prefix()
    if entity_data.get('feature_variant_sha256') != variant:
        raise ValueError('Entity gate feature variant mismatch')
    questions = [row['question'] for row in ref['rows']]
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                           add_generation_prompt=True) for q in questions]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                        return_offsets_mapping=True)
    features, masks = runtime.extract(model, tokenizer, questions, batch_size=1)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    locator = nn.Linear(vectors.shape[-1], 1).eval()
    locator.load_state_dict(router['locator_state'], strict=True)
    entity = nn.Linear(vectors.shape[-1], 1).eval()
    entity.load_state_dict(entity_data['gate_state'], strict=True)
    with torch.no_grad():
        locator_scores = locator(vectors).squeeze(-1).sigmoid()
        entity_scores = entity(pool(features, masks).float()).squeeze(-1).sigmoid()
    rows = []
    for i, question in enumerate(questions):
        if spans['rows'][i]['question'] != question:
            raise ValueError('Reference question order mismatch')
        selected, single = single_span_mask(locator_scores[i], masks[i], router['locator_threshold'])
        start = texts[i].find(question)
        chars = set()
        for token in selected.nonzero(as_tuple=True)[0].tolist():
            left, right = encoded['offset_mapping'][i, token].tolist()
            chars.update(range(max(0, left - start), min(len(question), right - start)))
        text = ''.join(question[j] for j in sorted(chars))
        reference_score = float(ref['rows'][i]['entity_score'])
        rows.append({'question': question, 'selected_text': text, 'single': single,
                     'reference_text': spans['rows'][i]['selected_text'],
                     'span_matches': text == spans['rows'][i]['selected_text'] and single == spans['rows'][i]['single'],
                     'entity_score': float(entity_scores[i]), 'reference_entity_score': reference_score,
                     'entity_absolute_difference': abs(float(entity_scores[i]) - reference_score)})
    result = {'status': 'reduced_prefix_development_parity', 'source_checkpoint_sha256': checkpoint_sha,
              'feature_variant_sha256': variant, 'rows': rows, 'span_matches': sum(r['span_matches'] for r in rows),
              'examples': len(rows), 'max_entity_score_difference': max(r['entity_absolute_difference'] for r in rows),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.perf_counter() - started,
              'limitations': ['Checks outputs of two heads on ten examples, not all internal feature elements.',
                              'Uses a reduced 21-layer prefix and does not perform generation or memory reads.',
                              'Saved full-model reference and example order are checked against checkpoint digest.',
                              'Parity on these cases does not prove equivalence for all inputs.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('examples', 'span_matches', 'max_entity_score_difference', 'peak_allocated_bytes')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
