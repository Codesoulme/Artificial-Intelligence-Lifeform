"""Test learned clarification and versioned neural reads in a two-turn dialogue."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension
from scripts.audit_spora_native_version_families import addresses, load_schedules
from scripts.audit_spora_version_reply_actions import atomic_reply
from scripts.audit_spora_version_roles import TRAIN_NAMES, clause_features, flat_features, role_predictions
from scripts.audit_spora_version_timeline import SUBJECTS, TODAY
from scripts.audit_spora_versioned_memory_generation import read_versions
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_clarification import Intent, resolve_version_turn, start_version_turn
from spora.research_version_roles import ABSTAIN, VersionRoleHead
from spora.research_version_scope import addressed_subject
from spora.research_version_timeline import effective_order
from spora.research_versioned_memory_pointer import VersionReplyActions


SEED = 25937
# Three version descriptions plus four actions. The field reference appears in every action.
TRAIN = (
    (('目前生效的一份', '最初落笔那一份', '现用前一份'), '拿{v}来。{q}记录写了什么？',
     '{q}记了什么我想核对，但请先和我确认具体版本。', '想核对{q}记录的事往后放，我叫你时再看。',
     '{q}写进故事标题合适吗？帮我看看语气。'),
    (('当下按的那份', '刚建档的底本', '前一版'), '我就查{v}。{q}是哪项？',
     '我只想要{q}其中一版，你得先问我是哪版。', '关于这件事，{q}先搁着，等我再开口。',
     '做个空白模板。{q}这一栏怎么排？'),
    (('眼下仍有效的', '开头录入的', '上一版'), '照着{v}核对。{q}是什么？',
     '我需要{q}的一份，时间范围我还没讲清，你先确认。', '核对可以以后再做。{q}这回先不用翻。',
     '我在写小说，{q}这个词组选得合适吗？'),
    (('当前版', '原始的首版', '紧邻现用前的那一版'), '按{v}来答。{q}写的是啥？',
     '{q}的记录有几个稿子，我只问其中一份，先让我说清是哪份。', '等我准备好再告诉你。{q}先不处理。',
     '只改下措辞。{q}怎么说更顺？'),
    (('现在用的登记', '第一次录下的那份', '现行前紧挨的旧份'), '核对{v}。{q}填的是什么？',
     '我想知道{q}的一条旧记载，哪一条还得你问我确认。', '别着急。{q}等下一轮再核对。',
     '帮忙做个栏目名，{q}这一行叫什么好？'),
    (('依旧作准的版本', '最早的一版', '刚被换下的一份'), '打开{v}核一下。{q}登记的是哪项？',
     '我问{q}的哪一版现在还不明确，麻烦先弄清楚。', '今天先缓一缓。{q}留到我下次提起。',
     '这是给作者的反馈。{q}这句读着别扭吗？'),
)
HELD = (
    (('现用版', '第一版', '紧挨现在的上一版'), '就取{v}。{q}记的是什么？',
     '只报一项就行。{q}我想找手上那张对应的一项，但具体哪份得先跟我对一下。',
     '把查阅留到我忙完。{q}现在缓办。', '只告诉我句子清不清楚。{q}记的是什么？'),
    (('手上还作数的那一条', '开档当天存入的一条', '新的一条换上来时退下的一条'),
     '帮我翻{v}。{q}记的是什么？', '我想核对一下{q}写的是哪项，但我说的那份有可能是新也可能是旧。',
     '先别动，{q}记的是什么这件事晚点再核。', '润色这句话。{q}记的是什么？'),
    (('这会儿一直沿用的', '还空着纸时填进去的开头一笔', '最近一回交班前尚有效的'),
     '查给我{v}。{q}是哪项？', '我没说准要的是哪一次登记。{q}写的是什么，先同我确认该看哪份。',
     '我还没决定查不查。{q}先别翻了。', '排个表格，不用填值。{q}这一格放哪合适？'),
)
FOLLOWUPS = (('现在这份。', 0, Intent.READ), ('就查第一版吧。', 1, Intent.READ), ('前一版即可。', 2, Intent.READ),
             ('还是算了。', None, Intent.DEFER), ('前一版或第一版。', None, Intent.OTHER),
             ('换个话题，说说第一版印刷史。', None, Intent.OTHER))


def examples(families, names):
    rows = []
    for phrases, read, ask, defer, other in families:
        for subject, (name, field) in enumerate(names):
            item = f'{name}的{field}'
            rows.extend({'question': read.format(q=item, v=phrase), 'subject': subject, 'action': Intent.READ, 'version': i}
                        for i, phrase in enumerate(phrases))
            rows.extend({'question': text.format(q=item), 'subject': subject, 'action': intent, 'version': None}
                        for text, intent in ((ask, Intent.ASK), (defer, Intent.DEFER), (other, Intent.OTHER)))
    return rows


def action_head(width):
    return nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, len(Intent)))


def train_head(head, vectors, rows):
    labels = torch.tensor([row['action'] for row in rows])
    weight = torch.bincount(labels, minlength=len(Intent)).float().reciprocal()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    rng = torch.Generator().manual_seed(SEED)
    for _ in range(200):
        indices = torch.randperm(len(rows), generator=rng)[:32]
        loss = F.cross_entropy(head(vectors[indices]), labels[indices], weight=weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print('Action head loss', float(loss.detach()), flush=True)


def decision_record(decision):
    return {'action': decision.action.name, 'route': list(decision.route) if decision.route is not None else None,
            'question': decision.pending.question if decision.pending is not None else None}


@torch.inference_mode()
def run_dialogues(model, tokenizer, head, bank, schedules, held, actions, plans):
    embedding = model.get_input_embeddings().weight.detach()
    codebook = F.normalize(embedding.index_select(1, compress_dimension(embedding.shape[1]).to(embedding.device)).float(), dim=-1)
    names, fields = addresses()
    cache = {}
    reads = []

    def generate(question, route, gold, variant, history=()):
        if route is None:
            return None
        subject, slot = route
        name = SUBJECTS[subject][1]
        if route not in cache:
            factor = bank['factors'][name][slot]
            cache[route] = read_versions({name: [factor]}, codebook)[name][0]
        result = atomic_reply(model, tokenizer, head, question, cache[route], history=history)
        hits = [[i, j] for i, (_, _, _, values, _) in enumerate(SUBJECTS) for j, value in enumerate(values)
                if value in result['answer']]
        correct = route == gold and hits == [list(gold)] and bool(result['token_ids']) and result['token_ids'][-1] == tokenizer.eos_token_id
        result.update({'hits': hits, 'correct': correct, 'variant': variant})
        reads.append(result)
        return result

    rows = []
    for fixture, action, plan in zip(held, actions, plans, strict=True):
        question, subject = fixture['question'], fixture['subject']
        address = addressed_subject(question, names, fields, colloquial=True)
        legacy_address = addressed_subject(question, names, fields)
        revisions = schedules.get(address, ())
        expected_order = effective_order(schedules[subject], as_of=TODAY, known_by=TODAY)
        coords = (expected_order[-1], expected_order[0], expected_order[-2])
        gold = None if fixture['version'] is None else (subject, coords[fixture['version']])
        baseline = start_version_turn(question, address, revisions, TODAY, ABSTAIN, plan['plan'], Intent.READ)
        candidate = start_version_turn(question, address, revisions, TODAY, ABSTAIN, plan['plan'], Intent(action))
        oracle = start_version_turn(question, address, revisions, TODAY, ABSTAIN, plan['plan'], fixture['action'])
        response = generate(question, candidate.route, gold, 'first')
        followups = []
        if fixture['action'] == Intent.ASK or candidate.pending is not None:
            for reply, reference, intent in FOLLOWUPS:
                continuation = resolve_version_turn(candidate.pending, reply, schedules, TODAY)
                expected = None if reference is None or fixture['action'] != Intent.ASK else (subject, coords[reference])
                history = () if candidate.pending is None else ({'role': 'user', 'content': question},
                           {'role': 'assistant', 'content': candidate.pending.question})
                generated = generate(reply, continuation.route, expected, 'followup', history)
                followups.append({'reply': reply, 'expected_action': (intent if fixture['action'] == Intent.ASK else Intent.OTHER).name,
                                  'expected_route': expected, **decision_record(continuation), 'response': generated})
        rows.append({**fixture, 'expected_route': gold, 'address': address, 'original_address': legacy_address,
                     'action_head': Intent(action).name, 'version_head': plan, 'baseline': decision_record(baseline),
                     'candidate': decision_record(candidate), 'oracle_action': decision_record(oracle),
                     'response': response, 'followups': followups})
        if len(rows) % 12 == 0:
            print('Replayed', len(rows), 'of', len(held), 'first turns', flush=True)
    return rows, reads


def summarize(rows, reads):
    def decision_correct(row, mode):
        return row[mode]['action'] == Intent(row['action']).name and (
            row[mode]['route'] == (None if row['expected_route'] is None else list(row['expected_route'])))

    decisions = {}
    for mode in ('baseline', 'candidate', 'oracle_action'):
        decisions[mode] = {}
        for kind in Intent:
            cases = [row for row in rows if row['action'] == kind]
            decisions[mode][kind.name] = {'total': len(cases), 'correct': sum(decision_correct(row, mode) for row in cases),
                                         'read': sum(row[mode]['route'] is not None for row in cases)}
    followups = [item for row in rows if row['action'] == Intent.ASK for item in row['followups']]
    expected_reads = [item for item in followups if item['expected_route'] is not None]
    return {'decisions': decisions, 'addressed': sum(row['address'] == row['subject'] for row in rows),
            'original_addressed': sum(row['original_address'] == row['subject'] for row in rows),
            'followup_reads': {'total': len(expected_reads), 'matched': sum(item['route'] == list(item['expected_route']) for item in expected_reads),
                               'correct_reply': sum(item['response'] is not None and item['response']['correct'] for item in expected_reads)},
            'cancel_or_other_wrong_reads': sum(item['route'] is not None for item in followups if item['expected_route'] is None),
            'all_generated': len(reads), 'correct_generated': sum(item['correct'] for item in reads)}


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    result_path, weights_path = args.output / 'result.json', args.output / 'action.pt'
    prior = json.loads(result_path.read_text(encoding='utf-8')) if args.verify else None
    sources = {str(path): base.sha(path) for path in (args.timeline, args.reader, args.version_heads,
               base.MODEL / 'model.safetensors', base.MODEL / 'tokenizer.json', base.MODEL / 'tokenizer_config.json')}
    if prior is not None and (sources != prior['sources'] or base.sha(weights_path) != prior['action_sha256']):
        raise ValueError('Replay input changed')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed()
    training = examples(TRAIN, TRAIN_NAMES)
    held = examples(HELD, [item[1:3] for item in SUBJECTS])
    questions = [row['question'] for row in training + held]
    if len(set(questions)) != len(questions):
        raise ValueError('Dialogue questions must be disjoint')
    action = action_head(model.config.hidden_size)
    if prior is None:
        train_head(action, flat_features(*clause_features(model, tokenizer, questions[:len(training)])), training)
        atomic_save(action.state_dict(), weights_path)
    else:
        action.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True), strict=True)
    vectors, mask = clause_features(model, tokenizer, questions[len(training):])
    with torch.inference_mode():
        actions = action.eval()(flat_features(vectors, mask)).argmax(-1).tolist()
    version = VersionRoleHead(model.config.hidden_size)
    version.load_state_dict(torch.load(args.version_heads, map_location='cpu', weights_only=True)['expanded']['weights'], strict=True)
    plans = role_predictions(version.eval(), vectors, mask)
    bank, schedules = load_schedules(args.timeline)
    reader_state = torch.load(args.reader, map_location='cpu', weights_only=True)['atomic']
    reader = VersionReplyActions(model.config.hidden_size, reader_state['scaffold_tokens'])
    reader.load_state_dict(reader_state, strict=True)
    rows, reads = run_dialogues(model, tokenizer, reader.eval().to('cuda'), bank, dict(enumerate(schedules)), held, actions, plans)
    if any(base.sha(Path(path)) != digest for path, digest in sources.items()):
        raise ValueError('A source changed during inference')
    stats = summarize(rows, reads)
    report = {'sources': sources, 'action_sha256': base.sha(weights_path), 'training': len(training), 'summary': stats, 'rows': rows}
    # Compare canonical JSON because round-tripping turns serializes tuples as lists.
    if prior is not None and json.loads(json.dumps(report)) != {key: prior[key] for key in report}:
        raise ValueError('Decisions or generated tokens changed after loading the checkpoint')
    store_json(args.output / ('reload.json' if args.verify else 'result.json'),
               {**report, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()})
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('timeline', 'reader', 'version_heads', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
