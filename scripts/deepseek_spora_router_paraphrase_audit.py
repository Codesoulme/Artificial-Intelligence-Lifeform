"""Compare router probes with more training paraphrases and new questions.

The learned gate is a research probe over frozen hidden states. Fact labels and
the two existing parameter branches are supplied by the experiment.
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
from scripts import deepseek_spora_internal_router_probe as probe
from scripts import deepseek_spora_multirelation_correction_audit as multi
from scripts import deepseek_spora_resume_equivalence_audit as io
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts import deepseek_spora_unknown_diversity_audit as branches_source
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_online_lora_memory_pilot import OnlineDelta
from spora import research_adapter_checkpoint as checkpoint

OUT = ROOT / 'runs/deepseek-spora-router-paraphrase-audit-20261116'
ADDED = {
    'drink': ('{name}平时偏爱什么饮品？', '请问{name}的饮料偏好是什么？',
              '记忆中{name}喜欢的饮料是哪一种？'),
    'city': ('{name}平时住在什么地方？', '请问{name}生活在哪座城市？',
             '记忆里{name}的住处是哪个城市？'),
    'hobby': ('{name}通常喜欢哪种活动？', '请问{name}有什么兴趣爱好？',
              '记忆里{name}的兴趣是什么？'),
}
FRESH = {'drink': '记录里的{name}平时会喝什么？',
         'city': '按照记录，{name}定居在哪座城市？',
         'hobby': '查阅记录，{name}业余喜欢做什么？'}
FRESH_NAMES = ('阿青', '小遥', '阿沛', '小阮')


def design(ordinary):
    training = probe.training_rows(ordinary)
    for (name, relation), label in ([(pair, 'known') for pair in multi.CORRECTED]
                                    + [(pair, 'unknown') for pair in multi.UNKNOWN_TRAIN + branches_source.EXTRA_PAIRS]):
        for template in ADDED[relation]:
            training.append({'label': label, 'question': template.format(name=name)})
    fresh = [{'group': 'known_fresh', 'name': name, 'relation': relation,
              'question': FRESH[relation].format(name=name), 'expected': value}
             for (name, relation), value in multi.CORRECTED.items()]
    fresh += [{'group': 'unknown_person_fresh', 'name': name, 'relation': relation,
               'question': FRESH[relation].format(name=name), 'expected': '我不知道'}
              for name in FRESH_NAMES for relation in FRESH]
    fresh += [{'group': 'unknown_relation_fresh', 'name': name, 'relation': relation,
               'question': FRESH[relation].format(name=name), 'expected': '我不知道'}
              for name, relation in multi.MISSING_KNOWN]
    if (len({x['question'] for x in training}) != len(training)
            or len({x['question'] for x in fresh}) != len(fresh)
            or {x['question'] for x in training} & {x['question'] for x in fresh}):
        raise ValueError('Duplicate or overlapping training and fresh questions')
    forbidden = ({name for name, _ in multi.CORRECTED} | {name for name, _ in multi.UNKNOWN_TRAIN}
                 | set(branches_source.EXTRA_TRAIN) | set(multi.AUDIT_NAMES) | set(branches_source.FRESH_HELD))
    if forbidden & set(FRESH_NAMES) or len(set(FRESH_NAMES)) != len(FRESH_NAMES):
        raise ValueError('Fresh person overlaps previous experiment')
    return training, fresh


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    old_report = json.loads((probe.OUT / 'result.json').read_text(encoding='utf-8'))
    previous = json.loads((branches_source.OUT / 'result.json').read_text(encoding='utf-8'))
    binding = json.loads((binding_source.OUT / 'binding.json').read_text(encoding='utf-8'))
    model_sha = base.sha(base.MODEL / 'model.safetensors')
    feature_sha, files = binding_source.binding()
    old_gate = torch.load(probe.OUT / 'probe.pt', map_location='cpu', weights_only=True)
    if (model_sha != base.WEIGHT_SHA or model_sha != old_report['base_sha256']
            or old_gate['base_sha256'] != model_sha or old_gate['feature_sha256'] != feature_sha
            or feature_sha != binding['feature_sha256'] or files != binding['files']):
        raise ValueError('Base or original probe binding mismatch')
    branch_data = {item['branch']: item for item in previous['branches']}
    for name in ('original', 'diverse'):
        if base.sha(branches_source.OUT / f'{name}.pt') != branch_data[name]['candidate_sha256']:
            raise ValueError('Candidate digest mismatch')
    ordinary, _ = data.load_reviewed_prompts()
    training, fresh = design(ordinary)
    previous_rows = branch_data['original']['rows']
    previous_diverse = branch_data['diverse']['rows']
    if any(a['question'] != b['question'] or a['group'] != b['group']
           for a, b in zip(previous_rows, previous_diverse)) or len(previous_rows) != len(previous_diverse):
        raise ValueError('Unmatched prior results')
    if {row['question'] for row in fresh} & {row['question'] for row in previous_rows}:
        raise ValueError('Fresh evaluation overlaps previous questions')
    torch.manual_seed(probe.SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)

    def representation(question):
        inputs = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        with torch.inference_mode():
            result = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
        return result[0, -1].float().cpu()

    questions = list(dict.fromkeys([row['question'] for row in training]
                                   + [row['question'] for row in previous_rows + fresh]))
    features = {}
    for index, question in enumerate(questions):
        features[question] = representation(question)
        if (index + 1) % 75 == 0:
            print(json.dumps({'represented': index + 1, 'total': len(questions)}), flush=True)
    base_fresh = {row['question']: pilot.generate(model, tokenizer, row['question']) for row in fresh}
    modules = []
    for layer in binding_source.LAYERS:
        mlp = model.model.layers[layer].mlp
        mlp.down_proj = OnlineDelta(mlp.down_proj, 8)
        modules.append(mlp.down_proj)
    fresh_answers = {}
    for name in ('original', 'diverse'):
        checkpoint.restore(branches_source.OUT / f'{name}.pt', modules, base_sha256=model_sha,
                           feature_sha256=feature_sha, layers=binding_source.LAYERS)
        fresh_answers[name] = {row['question']: pilot.generate(model, tokenizer, row['question']) for row in fresh}
        print(json.dumps({'branch': name, 'fresh_answers': len(fresh)}), flush=True)

    x = torch.stack([features[row['question']] for row in training])
    mean = x.mean(dim=0)
    scale = x.std(dim=0).clamp_min(.01)
    x = (x - mean) / scale
    targets = torch.tensor([probe.LABELS.index(row['label']) for row in training])
    torch.manual_seed(probe.SEED)
    gate = torch.nn.Linear(x.shape[1], len(probe.LABELS))
    optim = torch.optim.AdamW(gate.parameters(), lr=.01, weight_decay=.1)
    weights = torch.tensor([1 / (targets == index).sum().item() for index in range(len(probe.LABELS))])
    for _ in range(300):
        optim.zero_grad(set_to_none=True)
        loss = F.cross_entropy(gate(x), targets, weight=weights)
        loss.backward()
        optim.step()
    old_linear = torch.nn.Linear(x.shape[1], len(probe.LABELS))
    old_linear.load_state_dict(old_gate['gate'])
    prior_probe_rows = {row['question']: row for row in old_report['rows']}
    base_report = json.loads((binding_source.OUT / 'training.json').read_text(encoding='utf-8'))
    base_answers = {row['question']: row['no_adapter'] for row in base_report['states'][-1]['rows']}
    for row in old_report['rows']:
        if row['route'] == 'ordinary':
            base_answers[row['question']] = row['answer']
    base_answers.update(base_fresh)
    # Compute any base fallback not generated in the earlier probe.
    for module in modules:
        module.enabled = False
    rows = []
    with torch.no_grad():
        for index, left in enumerate(previous_rows + fresh):
            question = left['question']
            feature = features[question]
            old_route = probe.LABELS[int(old_linear((feature - old_gate['mean']) / old_gate['scale']).argmax())]
            new_route = probe.LABELS[int(gate((feature - mean) / scale).argmax())]
            if index < len(previous_rows):
                if old_route != prior_probe_rows[question]['route']:
                    raise ValueError('Original probe did not reproduce prior route')
                answers = {'known': left['answer'], 'unknown': previous_diverse[index]['answer']}
            else:
                answers = {'known': fresh_answers['original'][question], 'unknown': fresh_answers['diverse'][question]}
            if 'ordinary' in (old_route, new_route) and question not in base_answers:
                base_answers[question] = pilot.generate(model, tokenizer, question)
            if question in base_answers:
                answers['ordinary'] = base_answers[question]
            expected_route = 'known' if left['group'] == 'known_fresh' else probe.expected_route(left['group'])
            entry = {'group': left['group'], 'question': question, 'expected_route': expected_route,
                     'old_route': old_route, 'new_route': new_route, 'answers': answers}
            if 'expected' in left:
                entry['expected'] = left['expected']
                for which, route in (('old', old_route), ('new', new_route)):
                    entry[f'{which}_exact'] = answers[route]['answer'].strip().rstrip('。.!！ ') == left['expected']
            rows.append(entry)
    if base.sha(base.MODEL / 'model.safetensors') != model_sha:
        raise ValueError('Base changed')
    torch.save({'mean': mean, 'scale': scale, 'gate': gate.state_dict(), 'labels': probe.LABELS,
                'base_sha256': model_sha, 'feature_sha256': feature_sha}, OUT / 'augmented-probe.pt')
    report = {'base_sha256': model_sha, 'train_rows': len(training),
              'train_accuracy': float((gate(x).argmax(-1) == targets).float().mean()),
              'fresh_questions': fresh, 'rows': rows, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic labeled prompts; fresh wording is not a real-world benchmark.',
                              'Prior examples and branch choices have been observed in development.',
                              'Novel names may have appeared in base pretraining or earlier project work.',
                              'Separate probe and replayed branch outputs, not integrated runtime routing.',
                              'Greedy generation capped at 24 tokens; exact short-answer scoring.']}
    io.write_json(OUT / 'result.json', report)
    print(json.dumps({'train_accuracy': report['train_accuracy'], 'rows': len(rows),
                      'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
