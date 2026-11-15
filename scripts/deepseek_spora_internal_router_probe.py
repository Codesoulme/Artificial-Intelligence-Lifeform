"""Probe internal-representation routing between saved editing candidates.

This replays saved branch outputs; it does not install a runtime router or
inject fact answers. Labeled synthetic prompts are supplied for the probe.
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
from scripts import deepseek_spora_multirelation_correction_audit as multi
from scripts import deepseek_spora_resume_equivalence_audit as io
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts import deepseek_spora_unknown_diversity_audit as previous
from scripts.deepseek_spora_full_stream_loader import load_full_streamed

OUT = ROOT / 'runs/deepseek-spora-internal-router-probe-20261116'
LABELS = ('known', 'unknown', 'ordinary')
SEED = 17521


def training_rows(ordinary):
    rows = []
    for name, relation in multi.CORRECTED:
        for template in multi.TEMPLATES[relation]['train']:
            rows.append({'label': 'known', 'question': template.format(name=name)})
    for name, relation in multi.UNKNOWN_TRAIN + previous.EXTRA_PAIRS:
        for template in multi.TEMPLATES[relation]['train']:
            rows.append({'label': 'unknown', 'question': template.format(name=name)})
    rows.extend({'label': 'ordinary', 'question': item['prompt']} for item in ordinary)
    if len({row['question'] for row in rows}) != len(rows):
        raise ValueError('Duplicate probe training question')
    return rows


def expected_route(group):
    if group.startswith(('old_', 'new_')):
        return 'known'
    if group.startswith('unknown_'):
        return 'unknown'
    if group.startswith('ordinary_'):
        return 'ordinary'
    raise ValueError('Unknown evaluation group')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    branch_report = json.loads((previous.OUT / 'result.json').read_text(encoding='utf-8'))
    binding = json.loads((binding_source.OUT / 'binding.json').read_text(encoding='utf-8'))
    base_sha = base.sha(base.MODEL / 'model.safetensors')
    feature_sha, files = binding_source.binding()
    if (base_sha != base.WEIGHT_SHA or base_sha != branch_report['base_sha256']
            or feature_sha != binding['feature_sha256'] or files != binding['files']):
        raise ValueError('Base binding mismatch')
    branches = {branch['branch']: branch for branch in branch_report['branches']}
    for name in ('original', 'diverse'):
        if base.sha(previous.OUT / f'{name}.pt') != branches[name]['candidate_sha256']:
            raise ValueError('Candidate checksum mismatch')
    original = branches['original']['rows']
    diverse = branches['diverse']['rows']
    if len(original) != len(diverse) or any(a['question'] != b['question'] or a['group'] != b['group']
                                            for a, b in zip(original, diverse)):
        raise ValueError('Branch examples do not align')
    base_report = json.loads((binding_source.OUT / 'training.json').read_text(encoding='utf-8'))
    base_answers = {row['question']: row['no_adapter'] for row in base_report['states'][-1]['rows']}
    ordinary, _ = data.load_reviewed_prompts()
    train = training_rows(ordinary)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)

    def representation(question):
        inputs = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        with torch.inference_mode():
            hidden = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
        return hidden[0, -1].float().cpu()

    features = {}
    questions = list(dict.fromkeys([row['question'] for row in train] + [row['question'] for row in original]))
    for index, question in enumerate(questions):
        features[question] = representation(question)
        if (index + 1) % 50 == 0:
            print(json.dumps({'represented': index + 1, 'total': len(questions)}), flush=True)
    x_train = torch.stack([features[row['question']] for row in train])
    mean = x_train.mean(dim=0)
    scale = x_train.std(dim=0).clamp_min(0.01)
    x_train = (x_train - mean) / scale
    targets = torch.tensor([LABELS.index(row['label']) for row in train])
    gate = torch.nn.Linear(x_train.shape[1], len(LABELS))
    torch.manual_seed(SEED)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=.01, weight_decay=.1)
    weights = torch.tensor([1 / (targets == index).sum().item() for index in range(len(LABELS))])
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(gate(x_train), targets, weight=weights)
        loss.backward()
        optimizer.step()
    train_accuracy = float((gate(x_train).argmax(dim=-1) == targets).float().mean())
    rows = []
    with torch.no_grad():
        for left, right in zip(original, diverse):
            feature = (features[left['question']] - mean) / scale
            probabilities = F.softmax(gate(feature), -1)
            route = LABELS[int(probabilities.argmax())]
            expected = expected_route(left['group'])
            if route == 'ordinary':
                if left['question'] not in base_answers:
                    base_answers[left['question']] = pilot.generate(model, tokenizer, left['question'])
                answer = base_answers[left['question']]
            else:
                answer = left['answer'] if route == 'known' else right['answer']
            reference = base_answers.get(left['question'])
            row = {'group': left['group'], 'question': left['question'], 'expected_route': expected,
                   'route': route, 'probabilities': probabilities.tolist(), 'answer': answer,
                   'route_correct': route == expected}
            if 'expected' in left:
                row['expected'] = left['expected']
                row['exact'] = answer is not None and answer['answer'].strip().rstrip('。.!！ ') == left['expected']
            if expected == 'ordinary':
                row['same_base'] = answer is not None and answer['token_ids'] == reference['token_ids']
            rows.append(row)
    if base.sha(base.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base changed during probe')
    torch.save({'mean': mean, 'scale': scale, 'gate': gate.state_dict(), 'labels': LABELS,
                'base_sha256': base_sha, 'feature_sha256': feature_sha}, OUT / 'probe.pt')
    report = {'base_sha256': base_sha, 'source_branch_sha256': {name: branches[name]['candidate_sha256']
                                                                 for name in ('original', 'diverse')},
              'training': train, 'training_accuracy': train_accuracy, 'rows': rows,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Supervised three-way probe with synthetic labeled prompts.',
                              'Uses frozen base last-token representation, not a trained internal routing module.',
                              'Replays saved answers, not an integrated generation path.',
                              'Base fallback answers for misrouted memory questions are generated in this process.',
                              'Evaluation examples and branch choices were previously inspected.',
                              'Same eight ordinary training prompts overlap evaluation.',
                              'No claims about real-world identity, unseen relations, or independent chat quality.']}
    io.write_json(OUT / 'result.json', report)
    print(json.dumps({'training_accuracy': train_accuracy, 'rows': len(rows), 'seconds': report['seconds']}),
          flush=True)


if __name__ == '__main__':
    main()
