"""Read newly enrolled names before and after a committed SPORA memory correction."""

import argparse
import json
from pathlib import Path

from scripts.audit_qwen_compatible_natural_memory_20261121 import (
    MODEL, WORDS, closed_think_inputs, entity_query_proposer, load_head, select_event, sha, torch,
    _prepare_memory_read,
)
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_unknown_state_probe import fresh_controlled_state
from scripts.train_qwen_counterfactual_reply_20261122 import load_reply_head, sequence_processor
from spora.identity_fingerprints import IdentityFingerprints
from spora.integrated_durable import create_integrated_durable, load_integrated_durable
from spora.growing import query
from spora.research_temporal_intent import temporal_logits, time_selection
from transformers import AutoTokenizer


PEOPLE = (('小芮', 41), ('阿奕', 43))
FIRST, SECOND, CORRECTED = ('豆浆', '牛奶', '可乐')
QUESTIONS = (
    ('手头还有效的那页，给小芮备什么饮品？', 41, 0),
    ('查一下阿奕现用那份单子，他选的饮品是哪款？', 43, 0),
    ('记录更换以前，小芮原先的饮品是什么？', 41, 1),
    ('按刚改完的记录，小芮选的饮品是哪样？', 41, 2),
    ('照眼下那份清单，给小芮璇配哪款饮品？', None, None),
    ('小芮和阿奕各自登记了哪种饮品？两人的都告诉我。', None, None),
    ('先不要读取小芮的记录，给阿奕写句问候。', None, None),
)


def write_fact(runtime, key, word):
    decision, stats = runtime.observe_sourced(key, WORDS.index(word), 'direct', 1.)
    if not decision['current_write'] or stats['revision'] != runtime.store.revision:
        raise ValueError('Expected a committed current-value write')


def storage_control(runtime):
    """Check committed values independently from language routing."""
    selection = select_event(runtime.organism, runtime.state, PEOPLE[0][1], 'effective_previous',
                             as_of=runtime.state.cursor - 1)
    if selection['status'] != 'experimental_candidate':
        raise ValueError('Committed history is missing')
    device = next(runtime.organism.parameters()).device
    keys = torch.tensor([PEOPLE[0][1], PEOPLE[1][1], PEOPLE[0][1]], device=device)
    modes = torch.tensor([0, 0, 1], device=device)
    times = torch.tensor([0, 0, selection['event_time']], device=device)
    values = query(runtime.organism, runtime.state, keys, modes, times).argmax(-1).tolist()
    expected = [WORDS.index(word) for word in (CORRECTED, SECOND, FIRST)]
    if values != expected:
        raise ValueError('Committed current or previous value differs from the fixture')
    return {'values': [WORDS[value] for value in values], 'history_event': selection['event_time']}


def read_value(runtime, tokenizer, question, key, mode, reply=None):
    event = None
    if mode:
        intent = 'effective_previous' if mode == 1 else 'effective_latest'
        selection = select_event(runtime.organism, runtime.state, key, intent, as_of=runtime.state.cursor - 1)
        if selection['status'] != 'experimental_candidate':
            return {'event': None, 'value': None, 'text': None, 'reason': selection['reason']}
        event = selection['event_time']
    controls, evidence = _prepare_memory_read(runtime.organism, runtime.state, key,
                                              ('current', 'past', 'later')[mode], event)
    value = int(evidence.argmax(-1)[0])
    if value >= len(WORDS):
        return {'event': event, 'value': None, 'text': None, 'reason': 'unknown_value'}
    del controls['memory_state']
    controls['use_cache'] = True
    inputs = closed_think_inputs(tokenizer, question, 'cuda')
    length = inputs['input_ids'].shape[1]
    output_args = ({'memory_output_logits': evidence} if reply is None else
                   {'logits_processor': [sequence_processor(reply, evidence, length)]})
    output = runtime.generate(**inputs, **controls, **output_args, max_new_tokens=24, do_sample=False,
                              top_p=None, temperature=None, pad_token_id=tokenizer.pad_token_id,
                              eos_token_id=tokenizer.eos_token_id)
    tokens = output[0, length:].tolist()
    return {'event': event, 'value': WORDS[value], 'tokens': tokens,
            'text': tokenizer.decode(tokens, skip_special_tokens=True).strip(), 'reason': 'generated'}


@torch.no_grad()
def evaluate(runtime, tokenizer, router, temporal, *, corrected, reply=None):
    cases = QUESTIONS if corrected else QUESTIONS[:2]
    questions = [question for question, _, _ in cases]
    fingerprints = runtime.organism.identity_fingerprints
    if any(fingerprints.lookup(name) != key for name, key in PEOPLE):
        raise ValueError('Individual lost a registered test name')
    propose = entity_query_proposer(runtime.decoder, tokenizer, fingerprints, router, questions)
    features, mask = extract(runtime.decoder.base, tokenizer, questions, batch_size=1,
                             hidden_indices=tuple(temporal['hidden_indices']))
    probabilities = temporal_logits(temporal, features, mask).softmax(-1)
    modes = time_selection(probabilities, temporal['threshold'],
                           factorized=temporal.get('architecture') == 'factorized').tolist()
    expected_values = ((CORRECTED, SECOND, FIRST, CORRECTED, None, None, None) if corrected else (FIRST, SECOND))
    results = []
    for i, ((text, expected_key, expected_mode), mode, expected_value) in enumerate(zip(
            cases, modes, expected_values, strict=True)):
        key, confidence, reason, name = propose(i, text, None)
        actual = None
        if key is not None and mode != 3:
            actual = read_value(runtime, tokenizer, text, key, mode, reply)
        if expected_key is None:
            passed = actual is None
        else:
            passed = (key == expected_key and mode == expected_mode and actual is not None
                      and actual['value'] == expected_value and actual['text'] is not None
                      and expected_value in actual['text'])
        results.append({'question': text, 'expected_key': expected_key, 'expected_mode': expected_mode,
                        'expected_value': expected_value, 'key': key, 'mode': mode, 'name': name,
                        'confidence': confidence, 'reason': reason, 'read': actual, 'passed': passed})
    return results


def main(args):
    torch.set_num_threads(2)
    digest = sha(args.checkpoint)
    router, router_sha = load_head(args.router, 'router.pt', 'router_sha256', digest)
    temporal, temporal_sha = load_head(args.temporal, 'temporal.pt', 'temporal_sha256', digest)
    if router['format'] != 'spora-factorized-name-query-v1' or router['hidden_indices'] != temporal['hidden_indices']:
        raise ValueError('Expected compatible entity and temporal routers')
    folder = args.output
    database = folder / 'individual.sqlite'
    initial_path, reload_path = folder / 'before-reload.json', folder / 'after-reload.json'
    target = reload_path if args.reopen else initial_path
    if target.exists() or (not args.reopen and database.exists()):
        raise FileExistsError('Expected a new destination for this phase')
    sources = {'checkpoint_sha256': digest, 'router_sha256': router_sha, 'temporal_sha256': temporal_sha}
    reply = None
    if args.reply_head is not None:
        reply, sources['reply_sha256'] = load_reply_head(args.reply_head, digest, 'cuda')
    if args.reopen:
        initial = json.loads(initial_path.read_text(encoding='utf-8'))
        if initial['sources'] != sources:
            raise ValueError('Reopen must use the same checkpoint and router artifacts')
    folder.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    open_runtime = load_integrated_durable if args.reopen else create_integrated_durable
    with open_runtime(database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        if not args.reopen:
            runtime.state = fresh_controlled_state(runtime.organism)
            fingerprints = IdentityFingerprints(runtime.organism.config.key_count)
            for name, key in PEOPLE:
                fingerprints.bind(key, name)
            runtime.organism.identity_fingerprints = fingerprints
            for (_, key), value in zip(PEOPLE, (FIRST, SECOND), strict=True):
                write_fact(runtime, key, value)
            before = evaluate(runtime, tokenizer, router, temporal, corrected=False, reply=reply)
            write_fact(runtime, PEOPLE[0][1], CORRECTED)
        after = evaluate(runtime, tokenizer, router, temporal, corrected=True, reply=reply)
        control = storage_control(runtime)
        revision = runtime.store.revision
    if revision != 3 or sha(args.checkpoint) != digest:
        raise ValueError('Unexpected durable revision or modified bootstrap')
    report = {'sources': sources, 'revision': revision, 'after_correction': after, 'storage_control': control}
    if args.reopen:
        report['exact_reopen'] = initial['after_correction'] == after and initial['storage_control'] == control
    else:
        report['before_correction'] = before
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('revision', revision, 'passed', sum(row['passed'] for row in after), '/', len(after),
          'reopen parity', report.get('exact_reopen'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('checkpoint', 'router', 'temporal', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--reply-head', type=Path)
    parser.add_argument('--reopen', action='store_true')
    main(parser.parse_args())
