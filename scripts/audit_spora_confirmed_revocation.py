"""Train independent confirmed-address revocation and version probes."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_lookup_transitions import (
    ADD, HELD_PAIR, LookupTransitions, VERSION_TRANSFER, apply_operations, normalize_row, paired_rows,
    paired_scope_metrics, transition_features,
)
from scripts.audit_spora_recurrent_lookup_scope import SEED, episodes
from scripts.spora_version_chat import base, local_model
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save


# Each family changes a single instruction in the old-address clause.
REVOKE_TRAIN = (
    ('原先那项仍要核对', '原先那项不用核对'),
    ('请把前一个也留下', '请把前一个划掉'),
    ('刚刚选的照常计入', '刚刚选的从清单剔除'),
    ('头一项保留', '头一项撤销'),
    ('上一条继续处理', '上一条作废'),
    ('之前报的继续查', '之前报的不要再查'),
    ('先前那项仍有效', '先前那项已取消'),
    ('别删除原来的请求', '删除原来的请求'),
    ('把旧任务继续下去', '停掉旧任务'),
    ('原来定的维持不变', '原来定的已经撤回'),
)
REVOKE_CAL = (
    ('前面的查阅接着做', '前面的查阅到此为止'),
    ('头一个请求还成立', '头一个请求不成立'),
    ('原先报的继续有效', '原先报的失去效力'),
)
REVOKE_HELD = (
    ('起初选的那项也算数', '起初选的那项不算数'),
    ('照旧翻阅一开始说的', '放弃翻阅一开始说的'),
    ('前一笔暂时别拿掉', '前一笔先从这轮拿掉'),
    ('第一笔照样做完', '第一笔就此中止'),
    ('「旧的作废」只是举例，原来的照查', '刚才说原来的照查，现在让那条作废'),
    ('对方说不用查，我仍保留原先请求', '对方说照常查，我这边撤回原先请求'),
)
REVOKE_BLIND = (
    ('我说的第一件仍然算在内', '我说的第一件这回先排除'),
    ('新项加进来，原来的仍旧列着', '新项加进来，原来的从名单划去'),
    ('别把早先定的那件删了', '把早先定的那件从本轮剔除'),
    ('她说要撤掉，那只是她的意见，我还要查', '她说可以照查，那只是她的意见，我这边取消'),
    ('前一笔也在这次核查之内', '前一笔这次就先跳过'),
    ('照办我们最早提到的请求', '我们最早提到的请求不用执行了'),
)

VERSION_TRAIN = (
    ('第一回存进档的那份。', '我要初次登记的版本。', '从开头那次找。', '按最初留底的查。',
     '调出从前第一份。', '给我第一次记载的。', '只看最先入档的纪录。', '读一开始归档的内容。'),
    ('按照眼前生效的查。', '我现在该按哪一个？', '查目前管用的那个。', '把正在执行的一份调出来。',
     '请用现行的记录。', '我要此时适用那条。', '就查当下有效的版本。', '照现用的那份给我。'),
    ('把刚刚换掉的那条找出来。', '找在最新版本前面的那一项。', '我要前一份现行记录。', '查上次改掉的版本。',
     '来一份紧跟在现版之前的。', '回到最近一次撤下的那份。', '看现行那份的前身。', '上一回还用的记录。'),
    ('替我改写「我现在该按哪一个」。', '请翻译「查上次改掉的版本」。', '润饰「按最初留底的查」这句。',
     '请解释现行是什么意思。', '点评「第一回存进档的那份」是否通顺。', '我想了解档案怎么归类。',
     '帮我把「我要前一份」改得客气。', '把「请用现行的记录」缩成九个字。'),
)
VERSION_BLIND = (
    ('拿当年写进档案里的头一笔。', '请找档案建立时最先填的那项。', '用所有版本中最早留下的。'),
    ('现在对外采用的是哪条？', '这里该拿今天有效的那份。', '我就要正在适用的那项记录。'),
    ('给我现有那份的直接前一版。', '用这次改动覆盖掉的那项。', '取替换前最后有效的那份。'),
    ('这句「用所有版本中最早留下的」有歧义吗？', '帮我翻译「今天有效的那份」。', '解释一下「直接前一版」的含义。'),
)


def rows_from(groups):
    return [(phrase, label) for label, phrases in enumerate(groups) for phrase in phrases]


def clause_text(text):
    # Keep both sides of reported speech. Only ordinary delimiter punctuation separates evidence.
    return [part.strip() for part in re.split(r'[，,；;]', text) if part.strip()] or [text]


def encode_texts(extract, texts, *, clauses=False):
    views = [(text, *clause_text(text)) for text in texts] if clauses else [(text,) for text in texts]
    prompts = list(dict.fromkeys(text for group in views for text in group))
    vectors = {}
    for offset in range(0, len(prompts), 32):
        batch = prompts[offset:offset + 32]
        for text, hidden in zip(batch, extract(batch), strict=True):
            vectors[text] = hidden[:len(text)].float().mean(0)
    return torch.stack([torch.cat((vectors[group[0]], torch.stack([vectors[x] for x in group[1:]]).mean(0)))
                        if clauses else vectors[group[0]] for group in views])


def fit_probe(extract, examples, classes, *, clauses=False, encode=encode_texts):
    torch.manual_seed(SEED)
    values = encode(extract, [text for text, _ in examples], clauses=clauses)
    targets = torch.tensor([label for _, label in examples])
    head = nn.Sequential(nn.LayerNorm(values.shape[-1]), nn.Linear(values.shape[-1], 32), nn.GELU(), nn.Linear(32, classes))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    for _ in range(180):
        loss = F.cross_entropy(head(values), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


def revocation_examples(families):
    return [(phrase, label) for pair in families for label, phrase in enumerate(pair)]


@torch.inference_mode()
def classify(head, extract, examples, *, clauses=False, encode=encode_texts):
    features = encode(extract, [text for text, _ in examples], clauses=clauses)
    confidence, labels = head(features).softmax(-1).max(-1)
    return [dict(text=text, expected=target, predicted=guess, confidence=score)
            for (text, target), guess, score in zip(examples, labels.tolist(), confidence.tolist(), strict=True)]


def score(rows):
    counts = Counter()
    for row in rows:
        key = str(row['expected'])
        counts[f'{key}_total'] += 1
        counts[f'{key}_correct'] += row['predicted'] == row['expected']
    return dict(counts)


@torch.inference_mode()
def paired_replay(extract, baseline, revoke, *, threshold):
    pairs = paired_rows(episodes('held'), (HELD_PAIR,))
    rows = [row for pair in pairs for row in pair]
    data = transition_features(extract, {'held': rows}, utterance_features=True, clause_features=True)['held']
    _, history, valid = data
    known = torch.tensor([True, False]).expand(len(rows), -1)
    initial = baseline.advance(history, valid, opening=known)
    update = baseline.operation(history[:, 1]).argmax(-1)
    evidence = [normalize_row(row)[0].events[1] for row in rows]
    confidence, choice = revoke(encode_texts(extract, evidence, clauses=True)).softmax(-1).max(-1)
    # Apply the probe only to an explicit two-coordinate update with a confirmed starting item.
    # The baseline still has to select the second coordinate.
    applicable = update[:, 1].eq(ADD) & confidence.ge(threshold)
    final = apply_operations(known, update)
    final[:, 0] = torch.where(applicable, choice.eq(0), final[:, 0])
    revised = torch.stack((known, final), dim=1)
    return dict(total=len(pairs), baseline=paired_scope_metrics(initial, pairs),
                revised=paired_scope_metrics(revised, pairs), overrides=int(applicable.sum()),
                rows=[dict(request=row.events[1], question=row.question, expected=row.scope, active=state.tolist(),
                           baseline=old.tolist(), revocation=int(label), confidence=float(prob))
                      for row, state, old, label, prob in zip(rows, final, initial[:, -1], choice, confidence, strict=True)])


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    report = json.loads((args.baseline / 'result.json').read_text(encoding='utf-8'))
    old = args.baseline / 'clause.pt'
    old_sha = base.sha(old)
    if old_sha != report['results']['clause']['checkpoint_sha256']:
        raise ValueError('The clause checkpoint does not match its report')
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    source_paths = (args.recurrent[0], *(folder / 'attention.pt' for folder in args.recurrent[1:]))
    sources = {path: base.sha(path) for path in source_paths}
    baseline_sources = {Path(path).resolve(): digest for path, digest in report['sources'].items()}
    if any(baseline_sources.get(path.resolve()) != digest for path, digest in sources.items()):
        raise ValueError('The recurrent backbone differs from the clause experiment')
    with local_model(args.recurrent) as model:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
        width = model.config.hidden_size
        baseline = LookupTransitions(width, operation_width=2 * width).eval()
        baseline.load_state_dict(torch.load(old, map_location='cpu', weights_only=True), strict=True)
        train = [(f'{prefix}，{phrase}。', target) for prefix in ('查本项记录', '也查另一项记录', '请接着核对')
                 for phrase, target in revocation_examples(REVOKE_TRAIN)]
        revoke = fit_probe(extract, train, 2, clauses=True)
        calibration = classify(revoke, extract, revocation_examples(REVOKE_CAL), clauses=True)
        cutoffs = (.5, .65, .8, .95, 1.01)
        def objective(cutoff):
            right, wrong = 0, 0
            for row in calibration:
                if row['confidence'] >= cutoff:
                    right += row['predicted'] == row['expected']
                    wrong += row['predicted'] != row['expected']
            return right - 2 * wrong, -wrong, -cutoff

        cutoff = max(cutoffs, key=objective)
        version = fit_probe(extract, rows_from(VERSION_TRAIN), 4)
        for name, head in (('revocation', revoke), ('version', version)):
            path = args.output / f'{name}.pt'
            atomic_save(head.state_dict(), path)
            head.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        transfer = classify(revoke, extract, revocation_examples(REVOKE_HELD), clauses=True)
        blind_revoke = classify(revoke, extract, revocation_examples(REVOKE_BLIND), clauses=True)
        pair = paired_replay(extract, baseline, revoke, threshold=cutoff)
        version_test = classify(version, extract, rows_from(VERSION_TRANSFER))
        blind_version = classify(version, extract, rows_from(VERSION_BLIND))
    if base.sha(old) != old_sha or any(base.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('An input checkpoint changed')
    result = dict(revocation=score(transfer), version=score(version_test), revocation_rows=transfer,
                  version_rows=version_test, calibration=calibration, cutoff=cutoff, pairs=pair,
                  blind_revocation=score(blind_revoke), blind_version=score(blind_version),
                  blind_revocation_rows=blind_revoke, blind_version_rows=blind_version,
                  baseline_sha256=old_sha, sources={str(path): digest for path, digest in sources.items()},
                  seconds=perf_counter() - started)
    store_json(args.output / 'result.json', result)
    print('revocation', result['revocation'], 'version', result['version'],
          'paired', {key: pair[key] for key in ('baseline', 'revised', 'total', 'overrides')}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('baseline', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--recurrent', nargs=3, type=Path, required=True)
    main(cli.parse_args())
