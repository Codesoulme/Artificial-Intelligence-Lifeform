"""Compare joint-head predictions on reduced-prefix and recorded full-model features."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_joint_guard_audit import CANDIDATE
from scripts.deepseek_spora_joint_guard_diagnosis import OUTPUT as REFERENCE
from scripts.deepseek_spora_joint_span_pilot import JointSpan, selected_text
from scripts.deepseek_spora_prefix_features import load_prefix

OUTPUT = ROOT / 'runs/deepseek-spora-joint-prefix-parity-20261113.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    ref = json.loads(REFERENCE.read_text(encoding='utf-8'))
    source_sha = runtime.sha(runtime.CHECKPOINT)
    if ref['checkpoint_sha256'] != source_sha:
        raise ValueError('Checkpoint mismatch')
    path = CANDIDATE / 'span.pt'
    info = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    candidate_sha = runtime.sha(path)
    if candidate_sha != info['weights_sha256']:
        raise ValueError('Candidate mismatch')
    data = torch.load(path, map_location='cpu', weights_only=True)
    rows_ref = [row for row in ref['rows'] if row['results']['guarded']['guard_span'] is not None]
    questions = [row['question'] for row in rows_ref]
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, router, variant = load_prefix()
    if variant != data['feature_variant_sha256'] or variant != info['feature_variant_sha256']:
        raise ValueError('Feature variant mismatch')
    tokenizer = AutoTokenizer.from_pretrained(runtime.MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                           add_generation_prompt=True) for q in questions]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                        return_offsets_mapping=True)
    features, masks = runtime.extract(model, tokenizer, questions, batch_size=1)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    head = JointSpan(vectors.shape[-1], data['width']).eval()
    head.load_state_dict(data['state'], strict=True)
    rows = []
    with torch.no_grad():
        for i, row in enumerate(rows_ref):
            index = int(head(vectors[i], masks[i]).argmax())
            result = selected_text(index, encoded['offset_mapping'][i], texts[i].find(questions[i]), questions[i])
            full = row['results']['guarded']['guard_span']['joint']
            full_text = None if full is None else full['selected_text']
            rows.append({'question': questions[i], 'full_joint_text': full_text, 'prefix_joint_text': result,
                         'matches': full_text == result,
                         'full_abstained': full is None, 'prefix_abstained': result is None})
    if runtime.sha(path) != candidate_sha or runtime.sha(runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected input changed')
    result = {'status': 'development_only_head_prediction_parity', 'rows': rows,
              'examples': len(rows), 'matches': sum(r['matches'] for r in rows),
              'candidate_sha256': candidate_sha, 'source_checkpoint_sha256': source_sha,
              'feature_variant_sha256': variant, 'seconds': time.perf_counter() - started,
              'limitations': ['Only cases reaching the guard; same questions previously used for development.',
                              'Prediction equality does not prove hidden states or future examples are equivalent.',
                              'No candidate deployment, new memory reads, or training.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'examples': result['examples'], 'matches': result['matches']}), flush=True)


if __name__ == '__main__':
    main()
