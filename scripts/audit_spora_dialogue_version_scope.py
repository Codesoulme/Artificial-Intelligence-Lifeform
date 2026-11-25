"""Compare standalone, persistent and turn-scoped version lookup in live dialogue."""

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_dual_version_reply import FIELDS, score
from scripts.audit_spora_dual_version_routing import TODAY, feature_rows, reply_to_route
from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension
from scripts.audit_spora_version_normalizer import choices
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_multi_version_route import Lookup, LookupHead, corroborated_version, resolve_address
from spora.research_version_dialogue_scope import (advance_scope, dialogue_targets, pending_after_lookup,
                                                   pending_followup_targets)
from spora.research_version_store import VersionStore
from spora.research_versioned_memory_pointer import VersionReplyActions


CASES = {
    'same_question_drink': (
        ('named', '请查闻韶宁的饮品当前版。', ('闻韶宁/drinks/2',)),
        ('follow', '那上一版呢？', ('闻韶宁/drinks/1',)),
        ('follow', '那首版呢？', ('闻韶宁/drinks/0',)),
    ),
    'same_question_item': (
        ('named', '请查方吟晚的用品当前版。', ('方吟晚/items/1',)),
        ('follow', '那上一版呢？', ('方吟晚/items/0',)),
    ),
    'spoken_version': (
        ('named', '帮我查闻韶宁的饮品最新记录。', ('闻韶宁/drinks/2',)),
        ('follow', '那刚换下来的那份呢？', ('闻韶宁/drinks/1',)),
        ('follow', '那最早的记录呢？', ('闻韶宁/drinks/0',)),
    ),
    'retarget_to_other_field': (
        ('named', '查一下贾宁杉的用品首版。', ('贾宁杉/items/0',)),
        ('follow', '那见面地点当前版呢？', ('贾宁杉/places/1',)),
        ('follow', '那上一版呢？', ('贾宁杉/places/0',)),
    ),
    'paired_same_person': (
        ('named', '核对闻韶宁的饮品当前版、见面地点当前版。', ('闻韶宁/drinks/2', '闻韶宁/places/1')),
        ('follow', '两项都查首版。', ('闻韶宁/drinks/0', '闻韶宁/places/0')),
        ('follow', '那饮品上一版呢？', ('闻韶宁/drinks/1',)),
    ),
    'paired_unresolved': (
        ('named', '请查丁蔓庭的书目当前版、饮品当前版。', ('丁蔓庭/books/1', '丁蔓庭/drinks/1')),
        ('clarify', '那上一版呢？', (None,)),
        ('named', '查丁蔓庭的书目上一版。', ('丁蔓庭/books/0',)),
    ),
    'same_field_two_people': (
        ('named', '核对闻韶宁的饮品当前版；核对闻绍宁的饮品当前版。', ('闻韶宁/drinks/2', '闻绍宁/drinks/0')),
        ('clarify', '那饮品上一版呢？', (None,)),
        ('named', '只查闻韶宁的饮品上一版。', ('闻韶宁/drinks/1',)),
    ),
    'explicit_switch': (
        ('named', '请查闻韶宁的饮品当前版。', ('闻韶宁/drinks/2',)),
        ('named', '不是闻韶宁的饮品，改查闻绍宁的饮品当前版。', ('闻绍宁/drinks/0',)),
        ('clarify', '那上一版呢？', (None,)),
    ),
    'explicit_unknown_clears': (
        ('named', '请查闻韶宁的饮品当前版。', ('闻韶宁/drinks/2',)),
        ('clarify', '那闻韶蔚的饮品上一版呢？', (None,)),
        ('clarify', '那首版呢？', (None,)),
    ),
    'turn_change': (
        ('named', '请查方吟晚的见面地点当前版。', ('方吟晚/places/1',)),
        ('other', '换个问题，怎么把桌面收拾整齐？', ()),
        ('clarify', '那上一版呢？', (None,)),
    ),
    'post_cancel': (
        ('named', '核对叶昕卓的书目当前版。', ('叶昕卓/books/1',)),
        ('other', '先不查了，聊点别的。', ()),
        ('clarify', '那首版呢？', (None,)),
    ),
    'ordinary_boundary': (
        ('named', '请查许照岚的用品当前版。', ('许照岚/items/1',)),
        ('other', '怎么给泡好的茶降温？', ()),
        ('clarify', '那上一版呢？', (None,)),
    ),
    'quote_boundary': (
        ('named', '请查贾宁杉的用品当前版。', ('贾宁杉/items/1',)),
        ('other', '把「那上一版呢」翻译成英语。', ()),
        ('clarify', '那首版呢？', (None,)),
    ),
    'missing_second': (
        ('named', '请查闻韶宁的饮品当前版；查闻绍宁的饮品当前版。', ('闻韶宁/drinks/2', '闻绍宁/drinks/0')),
        ('clarify', '两项都查上一版。', ('闻韶宁/drinks/1', None)),
        ('clarify', '那首版呢？', (None,)),
    ),
    'unbound_pronoun': (
        ('named', '查方吟晚的用品当前版。', ('方吟晚/items/1',)),
        ('clarify', '那他的是哪一版？', (None,)),
        ('clarify', '那闻绍宁的当前版呢？', (None,)),
    ),
}
MODES = ('standalone', 'persistent', 'bounded')
PENDING_CONTROLS = {
    'pending_cannot_change_field': (
        ('named', '请查贾宁杉的用品当前版。', ('贾宁杉/items/1',)),
        ('clarify', '那改之前用过的旧版本是哪份呢？', (None,)),
        ('clarify', '那见面地点首版呢？', (None,)),
    ),
    'pending_consumed_by_second_unresolved': (
        ('named', '请查方吟晚的用品当前版。', ('方吟晚/items/1',)),
        ('clarify', '那改之前用过的旧版本是哪份呢？', (None,)),
        ('clarify', '那不记得的那一版是什么？', (None,)),
        ('clarify', '那首版呢？', (None,)),
    ),
    'pending_expires_after_turn_change': (
        ('named', '请查叶昕卓的书目当前版。', ('叶昕卓/books/1',)),
        ('clarify', '那改之前用过的旧版本是哪份呢？', (None,)),
        ('other', '换个问题，怎么修理卷边的书页？', ()),
        ('clarify', '那首版呢？', (None,)),
    ),
    'named_pending_can_answer_disambiguation': (
        ('clarify', '请查闻韶宁的饮品首版还是现行版？', (None,)),
        ('follow', '那首版呢？', ('闻韶宁/drinks/0',)),
    ),
}


class TemporalProposals:
    def __init__(self, extract, width, state):
        self.extract = extract
        self.head = LookupHead(width).eval()
        self.head.load_state_dict(state['weights'])
        self.cutoffs = state['cutoffs']
        self.cache = {}

    def __call__(self, evidence):
        unique = [text for text in dict.fromkeys(evidence) if text not in self.cache]
        if unique:
            logits = self.head(feature_rows(self.extract, unique))
            for text, label in zip(unique, choices(logits, *self.cutoffs).tolist(), strict=True):
                self.cache[text] = corroborated_version(Lookup(label), text)
        return [self.cache[text] for text in evidence]


def plan(question, catalog, enrolled, scope, propose, *, bounded, pending=None):
    targets = (dialogue_targets(question, enrolled, FIELDS, scope, bounded=bounded) if pending is None
               else pending_followup_targets(question, enrolled, FIELDS, pending))
    proposals = iter(propose([item.evidence for item in targets if item.evidence is not None and item.address is not None]))
    route, detail = [], []
    for item in targets:
        query = next(proposals) if item.evidence is not None and item.address is not None else item.query
        number, reason = resolve_address(catalog, item.address, query, TODAY)
        route.append(number)
        detail.append({'source': item.source, 'address': asdict(item.address) if item.address else None,
                       'query': query.kind if query else None, 'number': number, 'reason': reason})
    return targets, route, detail


@torch.inference_mode()
def evaluate(model, tokenizer, store, codebook, reader, propose, metadata, *, cases=CASES, modes=MODES, scope_swap=True):
    catalog = store.catalog()
    enrolled = {address.name for address in catalog}
    logs, totals = {}, {mode: Counter() for mode in modes}
    memorized = tuple(item['text'] for item in metadata.values())
    for name, sequence in cases.items():
        results = logs[name] = {}
        generated = {}
        for mode in modes:
            scope, pending, history, rows = (), None, [], []
            for group, question, expected in sequence:
                if any(value in question for value in memorized):
                    raise ValueError('A user message contains a stored value')
                known = [metadata[event]['number'] if event else None for event in expected]
                wanted = [metadata[event]['text'] for event in expected if event]
                foils = [item['text'] for key, item in metadata.items() if key not in expected]
                targets, route, detail = plan(question, catalog, enrolled, scope, propose, bounded=mode != 'persistent',
                                               pending=pending)
                cache_key = (tuple((message['role'], message['content']) for message in history), question, tuple(route))
                if cache_key not in generated:
                    generated[cache_key] = reply_to_route(model, tokenizer, store, codebook, reader, question, route,
                                                         history=history)
                selected, output = generated[cache_key]
                correct_read = bool(known and selected == known)
                correct_answer = correct_read and score(output, wanted, foils, tokenizer.eos_token_id)['all']
                withheld = bool(route and not selected)
                stats = totals[mode]
                stats.update({'turns': 1, 'expected_reads': group in ('named', 'follow'),
                              'correct_read': correct_read, 'correct_answer': correct_answer,
                              'follow_turns': group == 'follow', 'correct_follow_read': group == 'follow' and correct_read,
                              'correct_follow_answer': group == 'follow' and correct_answer,
                              'named_turns': group == 'named', 'correct_named_read': group == 'named' and correct_read,
                              'clarify_turns': group == 'clarify', 'clarified': group == 'clarify' and withheld,
                              'other_turns': group == 'other', 'wrong_read': bool(selected) and not correct_read,
                              'stored_value_repeated_without_read': not selected and any(value in output['answer'] for value in memorized)})
                previous = scope
                if mode != 'standalone':
                    scope = advance_scope(scope, targets, selected, persistent=mode == 'persistent')
                row = {'group': group, 'question': question, 'expected': known, 'visible_history': list(history),
                       'scope_before': [asdict(item) for item in previous], 'scope_after': [asdict(item) for item in scope],
                       'route': route, 'selected': selected, 'targets': detail, 'read_correct': correct_read,
                       'answer_correct': correct_answer, **output}
                if mode == 'bounded_pending':
                    before = pending
                    pending = pending_after_lookup(catalog, targets, selected, tuple(entry['reason'] for entry in detail))
                    row['pending_before'] = asdict(before) if before else None
                    row['pending_after'] = asdict(pending) if pending else None
                rows.append(row)
                history = [*history, {'role': 'user', 'content': question}, {'role': 'assistant', 'content': output['answer']}][-4:]
            results[mode] = rows
        print(name, 'tested', flush=True)

    ablation = []
    for name in (('same_question_drink', 'same_question_item') if scope_swap else ()):
        row = logs[name]['bounded'][1]
        selected, output = reply_to_route(model, tokenizer, store, codebook, reader, row['question'], row['route'])
        verified = bool(selected and selected == row['expected'])
        expected = [entry['text'] for entry in metadata.values() if entry['number'] in selected]
        foils = [entry['text'] for entry in metadata.values() if entry['number'] not in selected]
        ablation.append({'name': name, 'question': row['question'], 'selected': selected,
                         'scoped_correct': verified and score(output, expected, foils, tokenizer.eos_token_id)['all'], **output})
    return logs, {mode: dict(counts) for mode, counts in totals.items()}, ablation


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    version_db = args.routing / 'versions.sqlite3'
    prior_path, reader_path, normalizer_path = args.routing / 'result.json', args.reader / 'head.pt', args.normalizer / 'normalizers.pt'
    evidence_path = args.normalizer / 'guarded.json'
    protected = {str(path): base.sha(path) for path in (version_db, prior_path, reader_path, normalizer_path, evidence_path)}
    reference = json.loads(prior_path.read_text(encoding='utf-8'))
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    if (reference['database_sha256'] != protected[str(version_db)] or reference['sources'][str(reader_path)] != protected[str(reader_path)]
            or evidence['normalizer_sha256'] != protected[str(normalizer_path)]
            or any(base.sha(Path(path)) != digest for path, digest in reference['sources'].items())):
        raise ValueError('Saved versions or reader differs from the prior experiments')
    binding = sha256(json.dumps(reference['sources'], sort_keys=True).encode()).hexdigest()
    prior = json.loads((args.output / 'result.json').read_text(encoding='utf-8')) if args.verify else None
    if prior and prior['sources'] != protected:
        raise ValueError('Prior replay used different model or records')
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    embedding = model.get_input_embeddings().weight.detach()
    columns = compress_dimension(embedding.shape[1]).to(embedding.device)
    codebook = F.normalize(embedding.index_select(1, columns).float(), dim=-1)
    reader_state = torch.load(reader_path, map_location='cpu', weights_only=True)
    reader = VersionReplyActions(model.config.hidden_size, reader_state['scaffold_tokens'], slots=2)
    reader.load_state_dict(reader_state)
    reader.eval().to(model.device)
    settings = torch.load(normalizer_path, map_location='cpu', weights_only=True)['final_layer']
    extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=len(model.model.layers))
    propose = TemporalProposals(extract, model.config.hidden_size, settings)
    with VersionStore(version_db, binding) as store:
        modes = (*MODES, 'bounded_pending') if args.pending_probe else MODES
        rows, totals, ablation = evaluate(model, tokenizer, store, codebook, reader, propose, reference['metadata'], modes=modes)
        controls = None
        if args.pending_probe:
            control_rows, control_totals, _ = evaluate(model, tokenizer, store, codebook, reader, propose,
                                                       reference['metadata'], cases=PENDING_CONTROLS,
                                                       modes=('bounded', 'bounded_pending'), scope_swap=False)
            controls = {'rows': control_rows, 'totals': control_totals}
    if any(base.sha(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError('A protected model or state artifact changed')
    result = {'sources': protected,
              'rows': {name: {mode: logs[mode] for mode in MODES} for name, logs in rows.items()},
              'totals': {mode: totals[mode] for mode in MODES}, 'blank_history_scope_swap': ablation}
    if prior and any(prior[key] != value for key, value in result.items()):
        raise ValueError('Restart changed a replay output or address')
    run_stats = {'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()}
    store_json(args.output / ('reload.json' if args.verify else 'result.json'), {**result, **run_stats})
    if args.pending_probe:
        store_json(args.output / 'pending-probe.json', {**result, 'rows': rows, 'totals': totals,
                                                       'controls': controls, **run_stats})
        print('pending controls', controls['totals'], flush=True)
    print(totals, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('reader', 'routing', 'normalizer', 'output'):
        cli.add_argument(name, type=Path)
    cli.add_argument('--verify', action='store_true')
    cli.add_argument('--pending-probe', action='store_true')
    options = cli.parse_args()
    if options.pending_probe and not options.verify:
        cli.error('Replay the original experiment before evaluating one-turn clarification')
    main(options)
