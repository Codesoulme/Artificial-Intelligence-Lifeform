"""Frozen-feature joint-span pilot, research only; no real memory writes."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import pyarrow.parquet as parquet
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_cluener_locator_train import TRAIN, TRAIN_SHA
from scripts.deepseek_cluener_span_audit import SOURCE, SOURCE_SHA
from scripts.deepseek_spora_boundary_train_pilot import extract_names, digest
from scripts.deepseek_spora_conversation_locator_audit import select_names
from scripts.deepseek_spora_prefix_features import load_prefix
from scripts.deepseek_spora_prefix_parity_audit import OUTPUT as PARITY
from scripts.deepseek_learned_name_span import single_span_mask

OUTPUT = ROOT / 'runs/deepseek-spora-joint-span-pilot-20261113'


class JointSpan(nn.Module):
    """Score a start/end pair jointly, with an explicit abstention option."""
    def __init__(self, size, width=64):
        super().__init__()
        self.start = nn.Linear(size, width)
        self.end = nn.Linear(size, width)
        self.start_score = nn.Linear(width, 1)
        self.end_score = nn.Linear(width, 1)
        self.none = nn.Parameter(torch.zeros(()))

    def forward(self, vectors, allowed):
        a, b = self.start(vectors).tanh(), self.end(vectors).tanh()
        scores = self.start_score(a) + self.end_score(b).T + a @ b.T / (a.shape[-1] ** 0.5)
        positions = torch.arange(len(allowed))
        valid = allowed[:, None] & allowed[None, :] & (positions[:, None] <= positions[None, :])
        valid &= (positions[None, :] - positions[:, None]) < 6
        scores = scores.masked_fill(~valid, -1e9)
        return torch.cat((self.none.reshape(1), scores.flatten()))


def token_span(offsets, allowed, text_start, question, subject):
    if subject is None:
        return 0
    left = question.find(subject)
    if left < 0 or question.find(subject, left + 1) >= 0:
        raise ValueError('Ambiguous training subject')
    right = left + len(subject)
    matched = [i for i, ((a, b), ok) in enumerate(zip(offsets.tolist(), allowed.tolist()))
               if ok and a < text_start + right and b > text_start + left]
    if not matched:
        return None
    first, last = matched[0], matched[-1]
    if offsets[first, 0].item() != text_start + left or offsets[last, 1].item() != text_start + right:
        return None
    if last - first >= 6:
        return None
    return 1 + first * len(allowed) + last


def selected_text(index, offsets, start, question):
    if index == 0:
        return None
    size = len(offsets)
    left, right = divmod(index - 1, size)
    a, b = offsets[left, 0].item() - start, offsets[right, 1].item() - start
    return question[max(0, a):min(len(question), b)]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if runtime.sha(TRAIN) != TRAIN_SHA or runtime.sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Dataset digest mismatch')
    training, validation = parquet.read_table(TRAIN).to_pylist(), parquet.read_table(SOURCE).to_pylist()
    train_names = sorted(extract_names(training) - extract_names(validation), key=digest)[:64]
    eval_names = [name for _, name in select_names(validation, {r['text'] for r in training})]
    if len(train_names) != 64 or set(train_names) & set(eval_names):
        raise ValueError('Name split failure')
    train, evaluation = [], []
    for name in train_names:
        for subject, template in ((name, '请问{name}平时爱喝哪种饮品？'),
                                  (name + '子', '你知道{name}钟爱什么饮料吗？'),
                                  ('小' + name, '能告诉我{name}喜欢什么饮品吗？')):
            train.append({'question': template.format(name=subject), 'subject': subject, 'group': 'single'})
    for i in range(0, 32, 2):
        train.append({'question': f'{train_names[i]}和{train_names[i+1]}分别喜欢喝什么？',
                      'subject': None, 'group': 'multiple'})
    for question in ('今天能帮我做什么？', '推荐一种饮料。', '你记得什么事情？', '说一个故事吧。',
                     '我现在该喝什么？', '天气怎么样？', '记忆是什么？', '请介绍一下自己。'):
        train.append({'question': question, 'subject': None, 'group': 'none'})
    for name in eval_names:
        for group, subject in (('known', name), ('near_suffix', name + '月'), ('near_prefix', '阿' + name)):
            evaluation.append({'question': f'我想知道{subject}目前喜欢的饮品。', 'subject': subject,
                               'reference': name, 'group': group})
    for i in range(0, 16, 2):
        evaluation.append({'question': f'比较{eval_names[i]}与{eval_names[i+1]}偏爱的饮料。',
                           'subject': None, 'group': 'multiple'})
    for question in ('你能推荐饮品吗？', '我该怎么休息？', '你现在在想什么？', '有记录吗？',
                     '讲个笑话。', '什么是学习？', '我最喜欢什么？', '当前有什么信息？'):
        evaluation.append({'question': question, 'subject': None, 'group': 'none'})
    if {r['question'] for r in train} & {r['question'] for r in evaluation}:
        raise ValueError('Question overlap')
    source_sha = runtime.sha(runtime.CHECKPOINT)
    parity = json.loads(PARITY.read_text(encoding='utf-8'))
    if (parity['source_checkpoint_sha256'] != source_sha or parity['examples'] != 10
            or parity['span_matches'] != 10 or parity['max_entity_score_difference'] > 0.001):
        raise ValueError('Prefix parity evidence missing')
    torch.manual_seed(51387)
    torch.set_num_threads(2)
    started = time.perf_counter()
    base, router, variant = load_prefix()
    if variant != parity['feature_variant_sha256']:
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
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1).cpu()
    masks = masks.cpu()
    offsets = encoding['offset_mapping']
    del base, features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    labels = [token_span(offsets[i], masks[i], texts[i].find(row['question']), row['question'], row['subject'])
              for i, row in enumerate(records)]
    usable = [i for i in range(len(train)) if labels[i] is not None]
    head = JointSpan(vectors.shape[-1])
    old = nn.Linear(vectors.shape[-1], 1).eval()
    old.load_state_dict(router['locator_state'], strict=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0001, weight_decay=0.01)
    generator = torch.Generator().manual_seed(51387)
    losses = []
    for step in range(1200):
        index = usable[torch.randint(len(usable), (), generator=generator).item()]
        loss = F.cross_entropy(head(vectors[index], masks[index]).unsqueeze(0), torch.tensor([labels[index]]))
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, 299, 599, 1199):
            losses.append({'step': step + 1, 'loss': float(loss.detach())})
    head.eval()
    rows = []
    with torch.no_grad():
        for i, row in enumerate(evaluation, start=len(train)):
            index = int(head(vectors[i], masks[i]).argmax())
            start = texts[i].find(row['question'])
            predicted = selected_text(index, offsets[i], start, row['question'])
            selected, single = single_span_mask(old(vectors[i]).squeeze(-1).sigmoid(), masks[i],
                                               router['locator_threshold'])
            positions = selected.nonzero(as_tuple=True)[0].tolist()
            old_text = (selected_text(1 + positions[0] * len(masks[i]) + positions[-1],
                                      offsets[i], start, row['question']) if single and positions else None)
            rows.append({'group': row['group'], 'question_sha256': digest(row['question']),
                         'representable': labels[i] is not None,
                         'joint_exact': predicted == row['subject'], 'default_exact': old_text == row['subject'],
                         'joint_single': predicted is not None, 'default_single': old_text is not None,
                         'joint_matches_reference': predicted is not None and predicted == row.get('reference'),
                         'default_matches_reference': old_text is not None and old_text == row.get('reference')})
    summary = {}
    for group in ('known', 'near_suffix', 'near_prefix', 'multiple', 'none'):
        subset = [r for r in rows if r['group'] == group]
        summary[group] = {'total': len(subset), **{k: sum(r[k] for r in subset) for k in
                          ('representable', 'joint_exact', 'default_exact', 'joint_single', 'default_single',
                           'joint_matches_reference', 'default_matches_reference')}}
    if runtime.sha(runtime.CHECKPOINT) != source_sha:
        raise ValueError('Source checkpoint changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    torch.save({'state': head.state_dict(), 'feature_variant_sha256': variant, 'width': 64}, OUTPUT / 'span.pt')
    report = {'status': 'development_only_not_deployed', 'summary': summary, 'rows': rows, 'losses': losses,
              'train_names': len(train_names), 'train_records': len(train), 'usable_train_records': len(usable),
              'train_sha256': TRAIN_SHA, 'validation_sha256': SOURCE_SHA, 'source_checkpoint_sha256': source_sha,
              'feature_variant_sha256': variant, 'weights_sha256': runtime.sha(OUTPUT / 'span.pt'),
              'seconds': time.perf_counter() - started,
              'limitations': ['Synthetic questions; validation names and split were previously used for development.',
                              'No actual enrollment, memory read or language generation; near names are artificial.',
                              'Training and evaluation name strings differ, but subnames and pretraining overlap are unknown.',
                              'Reduced-prefix parity established on only ten other examples.',
                              'Multiple-person abstention assumes all such questions should be refused in this pilot.']}
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
