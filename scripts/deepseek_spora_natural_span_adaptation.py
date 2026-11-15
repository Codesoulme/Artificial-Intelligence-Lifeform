"""Fine-tune an isolated span head on weakly labeled natural dialogue.

Training labels derive from DuConv knowledge triples and are not human gold.
The reviewed development sample is never used for gradient updates.
"""
import copy
import json
import sys
import time
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from scripts import deepseek_spora_boundary_train_pilot as boundary
from scripts import deepseek_spora_duconv_dev_identity_audit as dev
from scripts import deepseek_spora_duconv_span_audit as natural
from scripts import deepseek_spora_name_extraction_audit as synthetic
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.audit_p236_duconv import ARCHIVE, ARCHIVE_SHA, conversation_key
from scripts.deepseek_spora_joint_span_pilot import JointSpan, selected_text, token_span
from scripts.deepseek_spora_prefix_features import load_prefix
from scripts.deepseek_spora_prefix_parity_audit import OUTPUT as PARITY
from spora.identity_fingerprints import IdentityFingerprints

OUT = ROOT / 'runs/deepseek-spora-natural-span-adaptation-20261116'
SINGLE_COUNT = 96
MULTI_COUNT = 24
STEPS = 1200
SEED = 71923


def training_schedule(more_multiple):
    groups = ('weak_single', 'weak_multiple', 'synthetic_none')
    return groups + ('weak_multiple',) if more_multiple else groups


def build_training(lines, reviewed):
    held_docs = {row['doc'] for row in reviewed}
    held_names = {name for row in reviewed for name in row['names']}
    held_names |= {row['subject'] for row in reviewed if row['subject']}
    held_names |= set(synthetic.ENROLLED)
    selected = natural.select_candidates(lines, held_docs, count=512)
    singles = []
    seen_docs = set()
    for row in selected:
        if row['name'] in held_names or any(name in row['text'] for name in held_names):
            continue
        singles.append({'question': row['text'], 'subject': row['name'], 'group': 'weak_single',
                        'doc': row['doc']})
        seen_docs.add(row['doc'])
        if len(singles) == SINGLE_COUNT:
            break
    if len(singles) != SINGLE_COUNT:
        raise ValueError('Insufficient disjoint single-name training conversations')
    candidates = []
    for raw in lines:
        item = json.loads(raw)
        conversation = item.get('conversation')
        if not isinstance(conversation, list) or not all(isinstance(x, str) for x in conversation):
            continue
        doc = conversation_key(conversation)
        if doc in held_docs or doc in seen_docs:
            continue
        names = natural.person_names(item.get('knowledge'))
        for turn, text in enumerate(conversation):
            question = text.replace(' ', '')
            mentions = [name for name in names if question.count(name) == 1]
            if (8 <= len(question) <= 100 and len(mentions) >= 2
                    and not any(name in question for name in held_names)):
                candidates.append({'question': question, 'subject': None, 'group': 'weak_multiple',
                                   'doc': doc, 'rank': dev.digest(doc + '\0' + str(turn))})
    multiples = []
    for row in sorted(candidates, key=lambda item: item['rank']):
        if row['doc'] in seen_docs:
            continue
        multiples.append(row)
        seen_docs.add(row['doc'])
        if len(multiples) == MULTI_COUNT:
            break
    if len(multiples) != MULTI_COUNT:
        raise ValueError('Insufficient disjoint multiple-name training conversations')
    negatives = [{'question': q, 'subject': None, 'group': 'synthetic_none', 'doc': None}
                 for q in boundary.NEGATIVE_TRAIN]
    training = singles + multiples + negatives
    evaluation_questions = {row['text'] for row in reviewed} | {row[1] for row in synthetic.CASES}
    if ({row['question'] for row in training} & evaluation_questions
            or len({row['question'] for row in training}) != len(training)):
        raise ValueError('Duplicate or overlapping training question')
    return training


def main(more_multiple=False):
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    reviewed = dev.reviewed_sample(dev.select_sample())
    if runtime.sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Source archive mismatch')
    with ZipFile(ARCHIVE) as source:
        lines = list(source.open('DuConv/train.txt'))
    training = build_training(lines, reviewed)
    evaluation = [{'question': row['text'], 'subject': row['subject'],
                   'group': 'natural_' + row['reviewed_group'], 'doc': row['doc']} for row in reviewed]
    evaluation += [{'question': question, 'subject': expected, 'group': 'synthetic_' + group, 'doc': None}
                   for group, question, expected in synthetic.CASES]
    source_sha = runtime.sha(runtime.CHECKPOINT)
    parity = json.loads(PARITY.read_text(encoding='utf-8'))
    if (parity['source_checkpoint_sha256'] != source_sha or parity['examples'] != 10
            or parity['span_matches'] != 10 or parity['max_entity_score_difference'] > .001):
        raise ValueError('Prefix parity evidence missing')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model, _, variant = load_prefix()
    if variant != parity['feature_variant_sha256']:
        raise ValueError('Feature variant mismatch')
    tokenizer = AutoTokenizer.from_pretrained(runtime.MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    records = training + evaluation
    questions = [row['question'] for row in records]
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                           add_generation_prompt=True) for question in questions]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                        return_offsets_mapping=True)
    features, masks = runtime.extract(model, tokenizer, questions, batch_size=1)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],))
                         for i in range(2)], dim=-1).cpu()
    masks = masks.cpu()
    del model, features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    targets = [token_span(encoded['offset_mapping'][i], masks[i], texts[i].find(row['question']),
                          row['question'], row['subject']) for i, row in enumerate(records)]
    groups = {group: [i for i, row in enumerate(training) if row['group'] == group and targets[i] is not None]
              for group in ('weak_single', 'weak_multiple', 'synthetic_none')}
    if any(not indices for indices in groups.values()):
        raise ValueError('A training group has no representable examples')
    source_dir = natural.CANDIDATES['crossed']
    source_report = json.loads((source_dir / 'result.json').read_text(encoding='utf-8'))
    source_path = source_dir / 'span.pt'
    parent_sha = runtime.sha(source_path)
    if parent_sha != source_report['weights_sha256'] or source_report['feature_variant_sha256'] != variant:
        raise ValueError('Parent span head mismatch')
    parent = torch.load(source_path, map_location='cpu', weights_only=True)
    if parent['feature_variant_sha256'] != variant:
        raise ValueError('Parent feature mismatch')
    baseline = JointSpan(vectors.shape[-1], width=parent['width']).eval()
    baseline.load_state_dict(parent['state'], strict=True)
    candidate = copy.deepcopy(baseline).train()
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=.00005, weight_decay=.01)
    generator = torch.Generator().manual_seed(SEED)
    group_names = training_schedule(more_multiple)
    loss_log = []
    for step in range(STEPS):
        group = group_names[step % len(group_names)]
        options = groups[group]
        index = options[torch.randint(len(options), (), generator=generator).item()]
        loss = F.cross_entropy(candidate(vectors[index], masks[index]).unsqueeze(0), torch.tensor([targets[index]]))
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite adaptation loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 300 == 0:
            loss_log.append({'step': step + 1, 'loss': float(loss.detach())})
            print(json.dumps(loss_log[-1]), flush=True)
    candidate.eval()
    fingerprints = IdentityFingerprints(len(synthetic.ENROLLED))
    for key, name in enumerate(synthetic.ENROLLED):
        fingerprints.bind(key, name)
    rows = []
    with torch.no_grad():
        for i, item in enumerate(evaluation, start=len(training)):
            start = texts[i].find(item['question'])
            row = {'group': item['group'], 'question_sha256': dev.digest(item['question']),
                   'doc_sha256': item['doc'], 'representable': targets[i] is not None}
            for label, head in (('baseline', baseline), ('candidate', candidate)):
                index = int(head(vectors[i], masks[i]).argmax())
                selected = selected_text(index, encoded['offset_mapping'][i], start, item['question'])
                result = {'correct': selected == item['subject'], 'selected': selected is not None}
                if item['group'].startswith('synthetic_'):
                    matched = fingerprints.lookup(selected) if selected else None
                    expected_key = (synthetic.ENROLLED.index(item['subject'])
                                    if item['subject'] in synthetic.ENROLLED else None)
                    result['route_correct'] = matched == expected_key
                    result['false_accept'] = matched is not None and expected_key is None
                row[label] = result
            rows.append(row)
    if runtime.sha(ARCHIVE) != ARCHIVE_SHA or runtime.sha(runtime.CHECKPOINT) != source_sha or runtime.sha(source_path) != parent_sha:
        raise ValueError('Protected input changed')
    target_path = OUT / 'span.pt'
    torch.save({'state': candidate.state_dict(), 'width': parent['width'], 'feature_variant_sha256': variant,
                'parent_sha256': parent_sha}, target_path)
    summary = {}
    for group in sorted({row['group'] for row in rows}):
        items = [row for row in rows if row['group'] == group]
        summary[group] = {'total': len(items), 'representable': sum(row['representable'] for row in items)}
        for label in ('baseline', 'candidate'):
            metrics = ('correct', 'selected', 'route_correct', 'false_accept')
            summary[group][label] = {key: sum(row[label].get(key, False) for row in items) for key in metrics}
    report = {'status': 'weak_label_development_only', 'summary': summary, 'rows': rows,
              'training': [{'group': row['group'], 'question_sha256': dev.digest(row['question']),
                            'doc_sha256': row['doc']} for row in training],
              'training_groups': {key: len(value) for key, value in groups.items()},
              'training_schedule': group_names,
              'losses': loss_log, 'steps': STEPS, 'archive_sha256': ARCHIVE_SHA,
              'reviewed_manifest_sha256': dev.MANIFEST_SHA, 'source_checkpoint_sha256': source_sha,
              'feature_variant_sha256': variant, 'parent_sha256': parent_sha,
              'candidate_sha256': runtime.sha(target_path), 'seconds': time.perf_counter() - started,
              'limitations': ['DuConv knowledge-derived labels are weak and may miss other people.',
                              'Synthetic no-person examples were reused from earlier development.',
                              'Evaluated reviewed development sample was already inspected, not sealed.',
                              'Exact names and dialogues excluded from training, but substrings and pretraining overlap may remain.',
                              'Multiple-person abstention is a diagnostic task, not a universal reading rule.',
                              'Frozen SPORA graft prefix, not the current low-rank correction candidate.',
                              'Explicit simulated fingerprints, no actual individual memory or automatic registration.',
                              'One seed, one fixed training budget and no independent human annotation.']}
    with (OUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--more-multiple']:
        OUT = ROOT / 'runs/deepseek-spora-natural-span-more-multiple-20261116'
        main(more_multiple=True)
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('Usage: script [--more-multiple]')
