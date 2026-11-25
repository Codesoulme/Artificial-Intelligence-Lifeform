"""Audit learned revision status on human-authored CrossWOZ dialogue prefixes."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter
from zipfile import ZipFile

from scripts.audit_spora_dialogue_evidence import POSITIONS, generate, load_spora_for_eval, store_json
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.deepseek_local_baseline import sha, torch
from scripts.spora_crosswoz_revision_annotations import ACCEPTED, CONTROLS
from scripts.train_spora_revision_sources import FORMAT, VIEWS, SourceStatusHead, active_scores, arguments, capture
from scripts.spora_revision_source_data import Status


ARCHIVE_SHA = 'b2b717990f6f0a1c98daaf5dc465e61d42f7de69c0d31e88e674275b1f34c94c'
CONTACT = re.compile(r'https?://|www\.|[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?<!\d)\d{7,}(?!\d)|\b0\d{2,3}-\d{7,8}\b', re.I)


def validation(path):
    if sha(path) != ARCHIVE_SHA:
        raise ValueError('Expected the pinned CrossWOZ validation package')
    with ZipFile(path) as archive:
        if archive.namelist() != ['val.json']:
            raise ValueError('Unexpected CrossWOZ archive layout')
        return json.loads(archive.read('val.json'))


def corpus_rows(corpus):
    rows = []
    for dialogue, turn, field, old, proposal in ACCEPTED:
        texts = corpus[dialogue]['messages']
        if [row['role'] for row in texts[turn:turn + 3]] != ['usr', 'sys', 'usr']:
            raise ValueError('Accepted proposal does not have the reviewed speaker order')
        source = ((turn, old), (turn + 1, proposal))
        for kind, turns, active in (('pending', (turn, turn + 1), (0,)),
                                    ('accepted', (turn, turn + 1, turn + 2), (1,))):
            rows.append(dict(group=kind, dialogue=dialogue, turns=turns, sources=source, active=active, field=field))
    for kind, dialogue, turns, sources, active, reason in CONTROLS:
        rows.append(dict(group=kind, dialogue=dialogue, turns=turns, sources=sources, active=active, reason=reason))
    for row in rows:
        turns = row['turns']
        if turns != tuple(sorted(set(turns))) or any(index not in turns for index, _ in row['sources']):
            raise ValueError('Reviewed source falls outside the supplied dialogue prefix')
        messages = corpus[row['dialogue']]['messages']
        row['messages'] = [(index, messages[index]['role'], messages[index]['content']) for index in turns]
        if any(CONTACT.search(text) for _, _, text in row['messages']):
            raise ValueError('Exclude turns containing contact details from the review')
        for index, word in row['sources']:
            if messages[index]['content'].count(word) != 1:
                raise ValueError(f'A reviewed evidence value is ambiguous: {row["dialogue"]}, {index}')
    return rows


def arrange(row):
    parts, bodies = [], {}
    offset = 0
    for index, speaker, text in row['messages']:
        prefix = '用户：' if speaker == 'usr' else '助手：'
        parts.append(prefix + text + '\n')
        begin = offset + len(prefix)
        offset += len(parts[-1])
        bodies[index] = (slice(begin, begin + len(text)), text)
    sources = []
    for index, word in row['sources']:
        body, text = bodies[index]
        start = body.start + text.index(word)
        sources.append((slice(start, start + len(word)), body, float(index != row['turns'][0])))
    roles = tuple(Status.ACTIVE if index in row['active'] else Status.MENTION for index in range(2))
    return ''.join(parts), sources, bodies[row['turns'][-1]][0], roles


@torch.no_grad()
def score_heads(rows, features, saved):
    by_view = {}
    before_acceptance = torch.tensor([index - 1 if row['group'] == 'accepted' else index
                                      for index, row in enumerate(rows)])
    for view in VIEWS:
        head = SourceStatusHead(features['source_spans'].shape[-1]).eval()
        head.load_state_dict(saved['heads'][view])
        subset = before_acceptance if view == 'no_latest' else slice(None)
        logits = head(*arguments(features, view, subset))
        forced = active_scores(logits).argmax(1).tolist()
        active_mask = (logits.argmax(-1) == Status.ACTIVE).tolist()
        details, group_counts = [], {}
        for row, picked, mask in zip(rows, forced, active_mask, strict=True):
            active = tuple(index for index, flag in enumerate(mask) if flag)
            gold = row['active']
            status_correct = active == gold
            counters = group_counts.setdefault(row['group'], Counter())
            counters.update(cases=1, status=int(status_correct),
                            forced_single=int(len(gold) == 1 and picked in gold),
                            false_added=len(set(active) - set(gold)), missed=len(set(gold) - set(active)))
            details.append(dict(dialogue=row['dialogue'], turns=row['turns'], group=row['group'], predicted=active,
                                gold=gold, forced=picked, status_correct=status_correct))
        pairs = list(zip(details[:2 * len(ACCEPTED):2], details[1:2 * len(ACCEPTED):2], strict=True))
        by_view[view] = dict(by_group=group_counts, rows=details,
                             complete_forced=sum(a['forced'] == 0 and b['forced'] == 1 for a, b in pairs),
                             complete_status=sum(a['status_correct'] and b['status_correct'] for a, b in pairs))
    return by_view


def grade(result, wanted, previous):
    text = result['answer'].strip(' \n，。！；："“”')
    return {**result, 'contains': wanted in text, 'stale': previous in text,
            'exact': text.replace(' ', '') == wanted.replace(' ', '') and result['eos']}


def answers(chat, rows):
    results, totals = [], {mode: Counter() for mode in ('dialogue', 'given_value')}
    positions = min(POSITIONS, chat.position_limit)
    budget = 32
    for row in rows:
        if row['group'] != 'accepted':
            continue
        old, wanted = (word for _, word in row['sources'])
        field = row['field']
        turns, *_ = arrange(row)
        prompts = {'dialogue': f'根据这段对话，只输出用户最后接受的方案的{field}。\n{turns}',
                   'given_value': f'用户最终接受的方案中，{field}是{wanted}。只输出这个值。'}
        outcomes = {}
        for mode, prompt in prompts.items():
            formatted = chat.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                                            add_generation_prompt=True) + '</think>\n\n'
            size = len(chat.tokenizer.encode(formatted, add_special_tokens=False))
            if size + budget > positions:
                raise ValueError('Response prompt would truncate reviewed dialogue evidence')
            reply = generate(chat.decoder, chat.tokenizer, [], prompt, positions, budget=budget)
            if reply['prompt_tokens'] != size or reply['retained'] != 0:
                raise ValueError('Unexpected response context layout')
            outcomes[mode] = grade(reply, wanted, old)
        for mode, value in outcomes.items():
            totals[mode].update({key: value[key] for key in ('contains', 'stale', 'exact', 'eos')})
        results.append(dict(dialogue=row['dialogue'], turns=row['turns'], field=field, expected=wanted, outcomes=outcomes))
        print('answer', row['dialogue'], {mode: value['exact'] for mode, value in outcomes.items()}, flush=True)
    return {'summary': totals, 'rows': results}


def main(args):
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    destination = target / 'result.json'
    if destination.exists():
        raise FileExistsError(destination)
    corpus = validation(args.archive)
    rows = corpus_rows(corpus)
    manifest_digest = sha256(json.dumps(rows, ensure_ascii=False).encode('utf-8')).hexdigest()
    if args.review:
        store_json(target / 'review.json', {'archive_sha256': ARCHIVE_SHA, 'annotations_sha256': manifest_digest,
                                            'rows': rows})
        print(Counter(row['group'] for row in rows), flush=True)
        return
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    chat = load_spora_for_eval(target)
    saved = torch.load(args.heads, map_location='cpu', weights_only=True)
    fingerprint = {key: value for key, value in chat.awake_individual.items() if key != 'memory_path'}
    trained = {key: value for key, value in saved['runtime'].items() if key != 'memory_path'}
    if trained != fingerprint or saved['format'] != FORMAT:
        raise ValueError('The source-status heads use a different checkpoint')
    longest = max(len(chat.tokenizer.encode(arrange(row)[0], add_special_tokens=False)) for row in rows)
    if longest > min(POSITIONS, chat.position_limit):
        raise ValueError('Status input would exceed the validated context limit')
    features = capture(FrozenTurnFeatures(chat.decoder, chat.tokenizer), rows, arrange=arrange)
    reviews = score_heads(rows, features, saved)
    print({view: (counts['complete_forced'], counts['complete_status'], counts['by_group'])
           for view, counts in reviews.items()}, flush=True)
    replies = answers(chat, rows)
    store_json(destination, {'archive_sha256': ARCHIVE_SHA, 'annotations_sha256': manifest_digest,
                             'heads_sha256': sha(args.heads), 'runtime': chat.awake_individual,
                             'counts': Counter(row['group'] for row in rows), 'source_status': reviews,
                             'answers': replies, 'seconds': perf_counter() - started,
                             'peak_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'heads', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--review', action='store_true')
    main(parser.parse_args())
