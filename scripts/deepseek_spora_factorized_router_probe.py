"""Factorized entity/relation routing probe with oracle subject spans.

Subject spans and the fact-presence matrix are supplied by the experiment.
No fact values are inserted into the prompt or output. This is not an
autonomous identity resolver or a deployed SPORA router.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_adapter_restart_audit as binding_source
from scripts import deepseek_spora_diverse_kl_correction_audit as data
from scripts import deepseek_spora_internal_router_probe as router
from scripts import deepseek_spora_multirelation_correction_audit as multi
from scripts import deepseek_spora_resume_equivalence_audit as io
from scripts import deepseek_spora_router_paraphrase_audit as prior
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts import deepseek_spora_unknown_diversity_audit as diversity
from scripts.deepseek_spora_full_stream_loader import load_full_streamed

OUT = ROOT / 'runs/deepseek-spora-factorized-router-probe-20261116'
NAMES = tuple(dict.fromkeys(name for name, _ in multi.CORRECTED))
RELATIONS = tuple(multi.TEMPLATES)
UNKNOWN = len(NAMES)
ORDINARY = len(RELATIONS)


def training_rows(ordinary):
    rows = []
    pairs = list(multi.CORRECTED) + list(multi.UNKNOWN_TRAIN + diversity.EXTRA_PAIRS)
    for name, relation in pairs:
        templates = multi.TEMPLATES[relation]['train'] + prior.ADDED[relation]
        for template in templates:
            rows.append({'question': template.format(name=name), 'name': name, 'relation': relation})
    rows.extend({'question': item['prompt'], 'name': None, 'relation': None} for item in ordinary)
    if len({row['question'] for row in rows}) != len(rows):
        raise ValueError('Duplicate training question')
    return rows


def span_indices(question, name, prompt, offsets):
    if not name or question.count(name) != 1 or prompt.count(question) != 1:
        raise ValueError('Subject or question is ambiguous')
    left = prompt.index(question) + question.index(name)
    right = left + len(name)
    indices = [index for index, (start, end) in enumerate(offsets) if start < right and end > left]
    if not indices or min(offsets[i][0] for i in indices) > left or max(offsets[i][1] for i in indices) < right:
        raise ValueError('Subject span is not covered by tokenizer offsets')
    return indices


def fit(features, target, classes, seed):
    mean = features.mean(dim=0)
    scale = features.std(dim=0).clamp_min(.01)
    normalized = (features - mean) / scale
    torch.manual_seed(seed)
    head = torch.nn.Linear(normalized.shape[1], classes)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.1)
    weights = torch.tensor([1 / (target == index).sum().item() for index in range(classes)])
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(normalized), target, weight=weights)
        loss.backward()
        optimizer.step()
    accuracy = float((head(normalized).argmax(-1) == target).float().mean())
    return {'mean': mean, 'scale': scale, 'head': head, 'training_accuracy': accuracy}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    previous = json.loads((prior.OUT / 'result.json').read_text(encoding='utf-8'))
    branch_report = json.loads((diversity.OUT / 'result.json').read_text(encoding='utf-8'))
    binding = json.loads((binding_source.OUT / 'binding.json').read_text(encoding='utf-8'))
    base_sha = base.sha(base.MODEL / 'model.safetensors')
    feature_sha, files = binding_source.binding()
    if (base_sha != base.WEIGHT_SHA or base_sha != previous['base_sha256']
            or feature_sha != binding['feature_sha256'] or files != binding['files']):
        raise ValueError('Base binding mismatch')
    for branch in branch_report['branches']:
        if base.sha(diversity.OUT / f"{branch['branch']}.pt") != branch['candidate_sha256']:
            raise ValueError('Prior candidate mismatch')
    ordinary, _ = data.load_reviewed_prompts()
    train = training_rows(ordinary)
    branch_rows = branch_report['branches'][0]['rows']
    eval_metadata = branch_rows + previous['fresh_questions']
    if len(eval_metadata) != len(previous['rows']) or any(
            a['question'] != b['question'] or a['group'] != b['group']
            for a, b in zip(eval_metadata, previous['rows'])):
        raise ValueError('Evaluation metadata does not match prior results')
    question_metadata = {}
    for row in train + eval_metadata:
        question = row['question']
        name = row.get('name')
        if question in question_metadata and question_metadata[question] != name:
            raise ValueError('Inconsistent subject annotation')
        question_metadata[question] = name
    torch.manual_seed(router.SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    features = {}
    for index, (question, name) in enumerate(question_metadata.items()):
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                                add_generation_prompt=True) + '</think>\n\n'
        encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded.pop('offset_mapping')[0].tolist()
        indices = span_indices(question, name, prompt, offsets) if name is not None else []
        with torch.inference_mode():
            hidden = model.model(**encoded.to('cuda'), use_cache=False, return_dict=True).last_hidden_state[0].float()
        features[question] = {'relation': hidden[-1].cpu(),
                              'entity': hidden[indices].mean(dim=0).cpu() if indices else None}
        if (index + 1) % 75 == 0:
            print(json.dumps({'represented': index + 1, 'total': len(question_metadata)}), flush=True)
    entity_rows = [row for row in train if row['name'] is not None]
    entity_features = torch.stack([features[row['question']]['entity'] for row in entity_rows])
    entity_labels = torch.tensor([NAMES.index(row['name']) if row['name'] in NAMES else UNKNOWN for row in entity_rows])
    relation_features = torch.stack([features[row['question']]['relation'] for row in train])
    relation_labels = torch.tensor([RELATIONS.index(row['relation']) if row['relation'] in RELATIONS else ORDINARY
                                    for row in train])
    entity = fit(entity_features, entity_labels, len(NAMES) + 1, router.SEED)
    relation = fit(relation_features, relation_labels, len(RELATIONS) + 1, router.SEED + 1)
    presence = torch.zeros(len(NAMES), len(RELATIONS), dtype=torch.bool)
    for name, kind in multi.CORRECTED:
        presence[NAMES.index(name), RELATIONS.index(kind)] = True
    base_answers = {row['question']: row['no_adapter'] for row in json.loads(
        (binding_source.OUT / 'training.json').read_text(encoding='utf-8'))['states'][-1]['rows']}
    rows = []
    with torch.no_grad():
        for meta, old in zip(eval_metadata, previous['rows']):
            question, name = meta['question'], meta.get('name')
            feature = features[question]
            r = int(relation['head']((feature['relation'] - relation['mean']) / relation['scale']).argmax())
            e = (int(entity['head']((feature['entity'] - entity['mean']) / entity['scale']).argmax())
                 if name is not None else UNKNOWN)
            route = ('ordinary' if r == ORDINARY else 'unknown' if e == UNKNOWN
                     else 'known' if bool(presence[e, r]) else 'unknown')
            expected_route = old['expected_route']
            answers = dict(old['answers'])
            if route == 'ordinary' and 'ordinary' not in answers:
                if question not in base_answers:
                    base_answers[question] = pilot.generate(model, tokenizer, question)
                answers['ordinary'] = base_answers[question]
            answer = answers[route]
            expected_entity = NAMES.index(name) if name in NAMES else UNKNOWN
            expected_relation = RELATIONS.index(meta['relation']) if meta.get('relation') in RELATIONS else ORDINARY
            row = {'group': meta['group'], 'question': question, 'name': name,
                   'expected_route': expected_route, 'route': route, 'answer': answer,
                   'entity_correct': e == expected_entity if name is not None else None,
                   'relation_correct': r == expected_relation,
                   'previous_route': old['new_route']}
            if 'expected' in meta:
                row['expected'] = meta['expected']
                row['exact'] = answer['answer'].strip().rstrip('。.!！ ') == meta['expected']
                row['previous_exact'] = old['new_exact']
            rows.append(row)
    if base.sha(base.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base changed')
    torch.save({'entity': {'mean': entity['mean'], 'scale': entity['scale'], 'state': entity['head'].state_dict()},
                'relation': {'mean': relation['mean'], 'scale': relation['scale'],
                             'state': relation['head'].state_dict()}, 'presence': presence,
                'names': NAMES, 'relations': RELATIONS, 'base_sha256': base_sha,
                'feature_sha256': feature_sha}, OUT / 'probe.pt')
    report = {'base_sha256': base_sha, 'training_rows': len(train),
              'entity_training_accuracy': entity['training_accuracy'],
              'relation_training_accuracy': relation['training_accuracy'], 'rows': rows,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Oracle subject span and hand-supplied fact-presence matrix.',
                              'Names and presence stored separately; no automatic update or identity proof.',
                              'Development prompts, branches, and answers are previously inspected.',
                              'Replays saved branch outputs; not an integrated runtime.',
                              'Small synthetic dataset and one training seed.',
                              'Strict short-answer scoring; greedy outputs capped at 24 tokens.']}
    io.write_json(OUT / 'result.json', report)
    print(json.dumps({'rows': len(rows), 'entity_training_accuracy': entity['training_accuracy'],
                      'relation_training_accuracy': relation['training_accuracy'],
                      'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
