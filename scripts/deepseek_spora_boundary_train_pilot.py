"""Train and compare a frozen-feature name locator; controlled development only."""
import copy
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
from scripts.deepseek_cluener_span_audit import SOURCE, SOURCE_SHA, character_mask, labeled_names
from scripts.deepseek_cluener_locator_train import TRAIN, TRAIN_SHA, token_gold
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_spora_cluener_locator_retrain import OUTPUT as PARENT
from scripts.deepseek_spora_conversation_locator_audit import select_names
from scripts.deepseek_spora_prefix_features import load_prefix
from scripts.deepseek_spora_prefix_parity_audit import OUTPUT as PARITY

OUTPUT = ROOT / 'runs/deepseek-spora-boundary-train-pilot-20261112'
TRAIN_TEMPLATES = ('请问{name}平时爱喝哪种饮品？', '你知道{name}钟爱什么饮料吗？')
NEGATIVE_TRAIN = ('今天能帮我做什么？', '推荐一种饮料。', '你记得什么事情？', '说一个故事吧。',
                  '我现在该喝什么？', '天气怎么样？', '记忆是什么？', '请介绍一下自己。')
NEGATIVE_EVAL = ('你能推荐饮品吗？', '我该怎么休息？', '你现在在想什么？', '有记录吗？',
                 '讲个笑话。', '什么是学习？', '我最喜欢什么？', '当前有什么信息？')


def extract_names(rows):
    found = set()
    for row in rows:
        text = row.get('text')
        if not isinstance(text, str):
            continue
        for start, end in labeled_names(row):
            name = text[start:end]
            if 2 <= len(name) <= 4 and all('\u4e00' <= char <= '\u9fff' for char in name):
                found.add(name)
    return found


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if runtime.sha(SOURCE) != SOURCE_SHA or runtime.sha(TRAIN) != TRAIN_SHA:
        raise ValueError('Dataset digest mismatch')
    validation = parquet.read_table(SOURCE).to_pylist()
    training = parquet.read_table(TRAIN).to_pylist()
    valid_texts = {r['text'] for r in validation}
    training = [r for r in training if r['text'] not in valid_texts]
    train_names = sorted(extract_names(training) - extract_names(validation), key=digest)[:32]
    eval_names = [name for _, name in select_names(validation, {r['text'] for r in training})]
    if len(train_names) != 32 or set(train_names) & set(eval_names):
        raise ValueError('Insufficient distinct train names or split contamination')
    # Train on varied names and question wording; eval uses a separate template.
    train = []
    for name in train_names:
        for template, subject in zip(TRAIN_TEMPLATES, (name, name + '子')):
            train.append({'question': template.format(name=subject), 'subjects': [subject], 'group': 'single'})
    train.extend({'question': q, 'subjects': [], 'group': 'none'} for q in NEGATIVE_TRAIN)
    for i in range(0, 16, 2):
        a, b = train_names[i:i + 2]
        train.append({'question': f'{a}和{b}分别喜欢喝什么？', 'subjects': [a, b], 'group': 'multiple'})
    evaluation = []
    for name in eval_names:
        for group, subject in (('known', name), ('near_suffix', name + '子'), ('near_prefix', '小' + name)):
            evaluation.append({'question': f'{subject}现在喜欢喝什么？', 'subjects': [subject],
                               'known': name, 'group': group})
    evaluation.extend({'question': q, 'subjects': [], 'group': 'none'} for q in NEGATIVE_EVAL)
    for i in range(0, 16, 2):
        a, b = eval_names[i:i + 2]
        evaluation.append({'question': f'比较{a}和{b}的饮品喜好。', 'subjects': [a, b], 'group': 'multiple'})
    if {r['question'] for r in train} & {r['question'] for r in evaluation}:
        raise ValueError('Training question overlap')
    parent_path = PARENT / 'locator.pt'
    parent_sha = runtime.sha(parent_path)
    report = json.loads((PARENT / 'result.json').read_text(encoding='utf-8'))
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    if parent_sha != report['locator_sha256']:
        raise ValueError('Parent locator digest mismatch')
    source_sha = runtime.sha(runtime.CHECKPOINT)
    parity = json.loads(PARITY.read_text(encoding='utf-8'))
    if (parity['source_checkpoint_sha256'] != source_sha or parity['span_matches'] != parity['examples']
            or parity['examples'] != 10 or parity['max_entity_score_difference'] > 0.001):
        raise ValueError('Insufficient prefix parity evidence')
    torch.manual_seed(9811)
    torch.set_num_threads(2)
    started = time.perf_counter()
    base, router, variant = load_prefix()
    if variant != parent['feature_variant_sha256'] or variant != parity['feature_variant_sha256']:
        raise ValueError('Feature provenance mismatch')
    tokenizer = AutoTokenizer.from_pretrained(runtime.MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    records = train + evaluation
    questions = [r['question'] for r in records]
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                           add_generation_prompt=True) for q in questions]
    encoding = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                         return_offsets_mapping=True)
    features, masks = runtime.extract(base, tokenizer, questions, batch_size=1)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    labels = torch.zeros_like(masks[:len(train)], dtype=torch.float32)
    usable = torch.ones(len(train), dtype=torch.bool)
    for i, item in enumerate(train):
        spans = []
        for subject in item['subjects']:
            pos = item['question'].find(subject)
            if pos < 0 or item['question'].find(subject, pos + 1) >= 0:
                raise ValueError('Training subject missing or ambiguous')
            spans.append((pos, pos + len(subject)))
        labels[i], representable, _ = token_gold(encoding['offset_mapping'][i], masks[i],
                                                 texts[i].find(item['question']), len(item['question']), spans)
        usable[i] = representable
    train_mask = masks[:len(train)] & usable[:, None]
    train_x = vectors[:len(train)][train_mask]
    train_y = labels[train_mask]
    old = nn.Linear(train_x.shape[-1], 1).eval()
    old.load_state_dict(parent['state'], strict=True)
    new = copy.deepcopy(old).train()
    optimizer = torch.optim.AdamW(new.parameters(), lr=0.00005, weight_decay=0.01)
    pos_weight = (train_y.numel() - train_y.sum()) / train_y.sum()
    losses = []
    for step in range(200):
        loss = F.binary_cross_entropy_with_logits(new(train_x).squeeze(-1), train_y, pos_weight=pos_weight)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, 49, 99, 199):
            losses.append({'step': step + 1, 'loss': float(loss.detach())})
    new.eval()
    with torch.no_grad():
        all_scores = [head(vectors[len(train):]).squeeze(-1).sigmoid() for head in (old, new)]
    rows = []
    for i, item in enumerate(evaluation):
        full_i = len(train) + i
        start = texts[full_i].find(item['question'])
        row = {'group': item['group'], 'question_sha256': digest(item['question'])}
        for label, scores in zip(('parent', 'candidate'), all_scores):
            selected, single = single_span_mask(scores[i], masks[full_i], parent['threshold'])
            chars = character_mask(encoding['offset_mapping'][full_i], selected, start, len(item['question']))
            picked = ''.join(item['question'][j] for j in chars.nonzero(as_tuple=True)[0].tolist())
            row[label + '_single'] = single
            row[label + '_exact'] = bool(single and len(item['subjects']) == 1 and picked == item['subjects'][0])
            row[label + '_match_known'] = bool(single and 'known' in item and picked == item['known'])
        rows.append(row)
    summary = {}
    for group in ('known', 'near_suffix', 'near_prefix', 'none', 'multiple'):
        subset = [r for r in rows if r['group'] == group]
        summary[group] = {'total': len(subset), **{key: sum(r[key] for r in subset)
                          for key in ('parent_single', 'candidate_single', 'parent_exact', 'candidate_exact',
                                      'parent_match_known', 'candidate_match_known')}}
    if runtime.sha(runtime.CHECKPOINT) != source_sha or runtime.sha(parent_path) != parent_sha:
        raise ValueError('Protected inputs changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    model_file = OUTPUT / 'locator.pt'
    torch.save({'format': 'spora-boundary-pilot-v1', 'state': new.state_dict(), 'threshold': parent['threshold'],
                'parent_sha256': parent_sha, 'feature_variant_sha256': variant}, model_file)
    result = {'status': 'development_only_no_deployment', 'summary': summary, 'rows': rows, 'losses': losses,
              'training_names': len(train_names), 'training_examples': len(train), 'usable_examples': int(usable.sum()),
              'parent_sha256': parent_sha, 'locator_sha256': runtime.sha(model_file), 'feature_variant_sha256': variant,
              'source_checkpoint_sha256': source_sha, 'train_sha256': TRAIN_SHA, 'validation_sha256': SOURCE_SHA,
              'seconds': time.perf_counter() - started,
              'limitations': ['Synthetic questions and a small set of names; evaluation is a development set used in prior probe.',
                              'Training names exclude validation names, but near-name variants and pretraining overlap are unknown.',
                              'Multi-person or no-person single-span rate is a proxy, not end-to-end memory risk.',
                              'Gold known-name comparison is an oracle, not autonomous enrollment or address selection.',
                              'No real individual, generation, actual memory read, or deployed runtime evaluated.',
                              'Frozen reduced prefix has only ten-example output parity with full model.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
