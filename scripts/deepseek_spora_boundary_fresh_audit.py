"""Compare the trained locator on a further held-out selection and wording."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import pyarrow.parquet as parquet
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_cluener_span_audit import SOURCE, SOURCE_SHA, character_mask, labeled_names, select_rows
from scripts.deepseek_cluener_locator_train import TRAIN, TRAIN_SHA
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_spora_boundary_train_pilot import OUTPUT as CANDIDATE
from scripts.deepseek_spora_cluener_dual_span_audit import select_fresh
from scripts.deepseek_spora_cluener_locator_retrain import OUTPUT as PARENT
from scripts.deepseek_spora_conversation_locator_audit import select_names
from scripts.deepseek_spora_prefix_features import load_prefix
from scripts.deepseek_spora_prefix_parity_audit import OUTPUT as PARITY

OUTPUT = ROOT / 'runs/deepseek-spora-boundary-fresh-audit-20261112.json'
NEGATIVES = ('推荐一款茶饮。', '我们聊点别的吧。', '目前有没有记录？', '你还记得哪些事？',
             '说说你最擅长什么。', '帮我想想晚饭吃什么。', '想学中文怎么做？', '什么时候休息？')


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if runtime.sha(SOURCE) != SOURCE_SHA or runtime.sha(TRAIN) != TRAIN_SHA:
        raise ValueError('Dataset digest mismatch')
    source = parquet.read_table(SOURCE).to_pylist()
    train_texts = {r['text'] for r in parquet.read_table(TRAIN).to_pylist()}
    previous = select_names(source, train_texts)
    prior, _ = select_rows(source)
    recent = select_fresh(source, prior)
    excluded = {r['text_sha256'] for r in prior} | {r['digest'] for r in recent} | {d for d, _ in previous}
    old_names = {n for _, n in previous}
    candidates = []
    for item in source:
        text = item.get('text')
        if not isinstance(text, str) or text in train_texts or digest(text) in excluded:
            continue
        spans = labeled_names(item)
        if len(spans) != 1:
            continue
        name = text[spans[0][0]:spans[0][1]]
        if name in old_names or not 2 <= len(name) <= 4 or any(not '\u4e00' <= c <= '\u9fff' for c in name):
            continue
        candidates.append((digest(text), name))
    names = []
    for source_digest, name in sorted(candidates):
        if name not in {n for _, n in names}:
            names.append((source_digest, name))
            if len(names) == 20:
                break
    if len(names) != 20:
        raise ValueError('Insufficient new names')
    cases = []
    for source_digest, name in names:
        for group, subject, question in (
            ('known', name, f'能说说{name}的饮料喜好吗？'),
            ('near_suffix', name + '子', f'你还记得{name}子爱喝什么吗？'),
            ('near_prefix', '小' + name, f'关于小{name}，记录过什么饮品？')):
            cases.append({'group': group, 'subject': subject, 'known': name, 'question': question,
                          'source_digest': source_digest})
    cases.extend({'group': 'none', 'question': q} for q in NEGATIVES)
    for i in range(0, 16, 2):
        a, b = names[i][1], names[i + 1][1]
        cases.append({'group': 'multiple', 'question': f'{a}与{b}喝的东西一样吗？'})
    parent_path, candidate_path = PARENT / 'locator.pt', CANDIDATE / 'locator.pt'
    parent_sha, candidate_sha = runtime.sha(parent_path), runtime.sha(candidate_path)
    parent_report = json.loads((PARENT / 'result.json').read_text(encoding='utf-8'))
    candidate_report = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    candidate = torch.load(candidate_path, map_location='cpu', weights_only=True)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    parity = json.loads(PARITY.read_text(encoding='utf-8'))
    if (parent_sha != parent_report['locator_sha256'] or candidate_sha != candidate_report['locator_sha256']
            or candidate['parent_sha256'] != parent_sha or candidate_report['source_checkpoint_sha256'] != source_sha
            or parity['source_checkpoint_sha256'] != source_sha or parity['span_matches'] != 10
            or parity['examples'] != 10 or parity['max_entity_score_difference'] > 0.001):
        raise ValueError('Provenance or parity mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    base, router, variant = load_prefix()
    if any(variant != x for x in (candidate['feature_variant_sha256'], parent['feature_variant_sha256'],
                                  parity['feature_variant_sha256'])):
        raise ValueError('Feature variant mismatch')
    tokenizer = AutoTokenizer.from_pretrained(runtime.MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    questions = [r['question'] for r in cases]
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                           add_generation_prompt=True) for q in questions]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                        return_offsets_mapping=True)
    features, masks = runtime.extract(base, tokenizer, questions, batch_size=1)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    heads = []
    for data in (parent, candidate):
        head = nn.Linear(vectors.shape[-1], 1).eval()
        head.load_state_dict(data['state'], strict=True)
        heads.append(head)
    with torch.no_grad():
        scores = [head(vectors).squeeze(-1).sigmoid() for head in heads]
    rows = []
    for i, case in enumerate(cases):
        question = case['question']
        start = texts[i].find(question)
        row = {'group': case['group'], 'source_digest': case.get('source_digest'), 'question_sha256': digest(question)}
        for label, values, data in zip(('parent', 'candidate'), scores, (parent, candidate)):
            mask, single = single_span_mask(values[i], masks[i], data['threshold'])
            chars = character_mask(encoded['offset_mapping'][i], mask, start, len(question))
            picked = ''.join(question[j] for j in chars.nonzero(as_tuple=True)[0].tolist())
            row[label + '_single'] = single
            row[label + '_exact'] = bool(single and 'subject' in case and picked == case['subject'])
            row[label + '_match_known'] = bool(single and 'known' in case and picked == case['known'])
        rows.append(row)
    summary = {}
    for group in ('known', 'near_suffix', 'near_prefix', 'none', 'multiple'):
        subset = [r for r in rows if r['group'] == group]
        summary[group] = {'total': len(subset), **{key: sum(r[key] for r in subset)
                          for key in ('parent_single', 'candidate_single', 'parent_exact', 'candidate_exact',
                                      'parent_match_known', 'candidate_match_known')}}
    if any(runtime.sha(path) != expected for path, expected in ((runtime.CHECKPOINT, source_sha),
           (parent_path, parent_sha), (candidate_path, candidate_sha))):
        raise ValueError('Protected input changed')
    result = {'status': 'fresh_controlled_span_audit', 'summary': summary, 'rows': rows,
              'candidate_sha256': candidate_sha, 'parent_sha256': parent_sha, 'feature_variant_sha256': variant,
              'source_checkpoint_sha256': source_sha, 'source_sha256': SOURCE_SHA, 'train_sha256': TRAIN_SHA,
              'seconds': time.perf_counter() - started,
              'limitations': ['New examples and names in validation split, but split was used for earlier development.',
                              'Few synthetic templates; wording differs by group.',
                              'Gold known name is used as oracle for match; no actual address, read, or answer evaluated.',
                              'Training excludes exact validation names but near variants and base pretraining overlap are unknown.',
                              'Small none and multi-person sets; one-span counts are not end-to-end routing outcomes.',
                              'Prefix parity with full model tested only on ten earlier examples.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
