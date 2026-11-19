"""Compare inherited-turn classifiers with fixed current-turn priority."""

import argparse
import time
from pathlib import Path

from deepseek_memory_attribute_span import KEYS, Route, batch, fit, route_action
from deepseek_memory_dialogue_context import OUT as CONTEXT_RUN, features_for, route
from deepseek_memory_field_transfer import OUT as TRANSFER_RUN
from deepseek_memory_time_place import HELD_PAIRS, TRAIN_PAIRS, paired_rows
from deepseek_memory_turn_priority import OUT as PRIORITY_RUN, combine, gate_probabilities
from deepseek_output_aligned_memory import MODEL, ROOT, SOURCE, fingerprint, load_integrated, read, sha, torch, write

OUT = ROOT / 'runs/deepseek-memory-inherited-turn'
NOVEL = (
    ('说的是小林一直觉得好喝的那款。', '能说出具体是哪样吗？', 'drink', 0),
    ('他每周到市场做的活计有个专门名称。', '叫什么来着？', 'work', 0),
    ('我在对照登记表上那一站的站名。', '帮我念一下行吗？', 'place', 0),
    ('我想核实对方登记进门的时分。', '你查到这一项了吗？', 'time', 0),
    ('那阵子她出门登记的处所我知道，我想了解出发的日子。', '这一格怎么填？', 'time', 0),
    ('这张票的实付金额是上面要查的。', '那里具体写多少？', 'amount', 0),
    ('笔记提到了带着他一起上山的那个人。', '能说说是谁么？', 'other', 0),
    ('说的是程女士到过哪家博物馆。', '她在工作之余最爱点哪一种茶？', 'drink', 1),
    ('这会儿说的是她在哪儿上班。', '请写一条给朋友的生日祝词。', 'ordinary', 1),
    ('我记得他提过平常喝什么。', '请看看那阵子他究竟在哪个岗位任职。', 'work', 1),
    ('上面提到的是出门的地方。', '要看的是记录里他是何日回来的。', 'time', 1),
    ('回顾她日常的职业经历。', '这次改问她那趟行程落到哪个城市。', 'place', 1),
)


def previous_features(decoder, tokenizer, rows):
    return {row['key']: features_for(decoder, tokenizer, dict(question=row['history']), False) for row in rows}


def restrict_to_previous(features):
    masked = {}
    for key, (tail, tokens, labels, ids, turns) in features.items():
        previous = torch.tensor(turns) == 0
        masked[key] = tail, tokens[previous], labels[previous], ids, [0] * int(previous.sum())
    return masked


def evaluate(decoder, tokenizer, head, router, direct, gate, setup, rows):
    full = {r['key']: features_for(decoder, tokenizer, r, True) for r in rows}
    current = {r['key']: features_for(decoder, tokenizer, r, False) for r in rows}
    probability = gate_probabilities(gate, rows, current)
    explicit = route(direct, rows, current)
    candidates = {'original': route(router, rows, full),
                  'masked': route(router, rows, restrict_to_previous(full)),
                  'isolated': route(head, rows, previous_features(decoder, tokenizer, rows))}
    results = {name: combine(inherited, explicit, probability, setup['gate_cutoff'])
               for name, inherited in candidates.items()}
    results['agreement'] = [dict(old, agrees=not old['inherits'] or old['selected'] == other['selected'])
                            for old, other in zip(results['original'], results['isolated'], strict=True)]
    for group in results.values():
        for row in group:
            action = route_action(row, setup['cutoffs']['priority'])
            row['action'] = 'clarify' if action in ('read', 'ordinary') and not row.get('agrees', True) else action
    return results


def main(out):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    torch.set_num_threads(2)
    artifacts = (PRIORITY_RUN / 'plan.json', PRIORITY_RUN / 'routes.json', PRIORITY_RUN / 'routing.pt',
                 CONTEXT_RUN / 'router.pt', TRANSFER_RUN / 'plan.json')
    for path in artifacts:
        if sha(path) != read(path.parent / 'result.json')['artifact_sha256'][path.name]:
            raise ValueError(f'Source artifact changed: {path}')
    plan = read(PRIORITY_RUN / 'plan.json')
    setup = read(PRIORITY_RUN / 'routes.json')
    train = [r for r in plan['train'] if r['kind'] == 'follow'] + paired_rows('train', TRAIN_PAIRS)
    suites = {'calibration': plan['calibration'], 'prior': plan['held'], 'paired': paired_rows('paired', HELD_PAIRS),
              'transfer': read(TRANSFER_RUN / 'plan.json')['rows'],
              'novel': [dict(key=f'novel-{i}', history=h, question=q, field=f, target_turn=t)
                        for i, (h, q, f, t) in enumerate(NOVEL)]}
    if {r['history'] for r in train} & {r['history'] for group in suites.values() for r in group}:
        raise ValueError('A training history occurs in the evaluation')
    hashes = fingerprint((Path(__file__), SOURCE, *artifacts, *(MODEL / name for name in
                         ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'))))
    out.mkdir()
    write(out / 'plan.json', dict(train=train, suites=suites, sources=hashes))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
              dtype=torch.float16, attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, organism, state = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    del organism, state
    decoder.eval().requires_grad_(False)
    width = base.lm_head.weight.shape[1]
    old, direct = (Route(width, localized=True).eval() for _ in range(2))
    old.load_state_dict(torch.load(CONTEXT_RUN / 'router.pt', weights_only=True))
    priority = torch.load(PRIORITY_RUN / 'routing.pt', weights_only=True)
    direct.load_state_dict(priority['direct'])
    gate = torch.nn.Linear(width, 1).eval()
    gate.load_state_dict(priority['inheritance'])
    inputs = batch([dict(question=r['key']) for r in train], previous_features(decoder, tokenizer, train))
    head = fit(inputs, torch.tensor([KEYS.index(r['field']) for r in train]), localized=True)
    torch.save(head.state_dict(), out / 'previous-router.pt')
    head.load_state_dict(torch.load(out / 'previous-router.pt', weights_only=True))
    results = {}
    for name, group in suites.items():
        results[name] = evaluate(decoder, tokenizer, head, old, direct, gate, setup, group)
        print(f'Evaluated {name}: {len(group)} conversations', flush=True)
    if fingerprint(tuple(Path(path) for path in hashes)) != hashes:
        raise ValueError('An input changed during evaluation')
    write(out / 'result.json', dict(suites=results, cutoff=setup['cutoffs']['priority'],
          elapsed_seconds=time.perf_counter() - started, artifact_sha256={name: sha(out / name) for name in
          ('plan.json', 'previous-router.pt')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    main(parser.parse_args().out)
