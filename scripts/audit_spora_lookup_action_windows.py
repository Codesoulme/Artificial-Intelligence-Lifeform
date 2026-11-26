"""Test whether local activation pooling helps distinguish direct lookup actions."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from scripts.audit_spora_confirmed_revocation import classify, encode_texts, fit_probe
from scripts.audit_spora_lookup_confirmation import (
    CANCEL, CLARIFY, PROCEED, DIRECT_CALIBRATION, DIRECT_TRAIN, EXPLICIT, SPEAKER_CALIBRATION,
    SPEAKER_TRAIN, calibrate, decision, evaluate_choices, speaker_examples,
)
from scripts.audit_spora_recurrent_lookup_scope import PEOPLE
from scripts.spora_version_chat import FrozenTurnFeatures, VersionReplyActions, VersionStore, artifacts, base, compress_dimension, local_model
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save
from spora.research_version_store import MemoryAddress


TRAIN = (
    ('现在请{verb}整批栏目的核对。', '停止', '完成'),
    ('把手上全部检索{verb}。', '中止', '做完'),
    ('查阅清单里所有事项都要{verb}。', '撤下', '保留'),
    ('请{verb}读取剩下的资料。', '别再', '接着'),
    ('这次需要{verb}查下去了。', '停止', '继续'),
    ('我明确要求{verb}所有调取。', '终止', '维持'),
    ('对方想照办，我决定{verb}这些查询。', '撤销', '保留'),
    ('对方说放弃全部查阅，我要{verb}。', '放弃', '照办'),
)
CALIBRATION = (
    ('把所有已经安排的查询{verb}吧。', '叫停', '办妥'),
    ('当前这一批调阅{verb}。', '不必再做', '不许停做'),
    ('她建议作废，实际我决定{verb}整单。', '作废', '继续'),
)
HELD = {
    'verb': (
        ('请把这一轮所有检索{verb}。', '收掉', '办完'),
        ('待查清单上的项目统统{verb}。', '撤回', '执行'),
        ('我现在确认整批调档{verb}。', '终结', '照常'),
        ('手头每条查阅指令都请{verb}。', '叫停', '完成'),
    ),
    'negation': (
        ('这批记录{verb}继续查。', '一条都不必', '一条都得'),
        ('所有核验{verb}。', '不要继续', '不要停止'),
        ('本次调档{verb}。', '无须接着办', '无须中断'),
        ('剩下的事项都{verb}。', '不再核查', '不得漏查'),
    ),
    'speaker': (
        ('搭档建议暂停查阅，但我决定把整份清单{verb}。', '取消', '照办'),
        ('「接着查」是别人说的，我的要求是{verb}。', '不查了', '照常查'),
        ('别人要求把整批记录看完，我本人要求{verb}。', '停办', '继续'),
        ('同事说不该终止调阅，最后我决定{verb}。', '全部终止', '全部保留'),
    ),
}
UNCLEAR = (
    '「全部核验停办」是我转述的内容，我自己还没决定。',
    '有人希望照办，有人反对，我还没有给你指令。',
    '这一批档案要停还是要继续，先让我考虑。',
    '听到了你的提议，但先别替我作决定。',
)
OPTIONS = ('继续当前查阅', '取消本轮全部查阅', '用户尚未决定，需要澄清')


def action_pairs(families):
    return [(frame.format(verb=verb), label)
            for frame, stop, proceed in families for verb, label in ((stop, CANCEL), (proceed, PROCEED))]


def window_features(extract, texts, *, clauses=False):
    vectors = []
    for begin in range(0, len(texts), 16):
        batch = texts[begin:begin + 16]
        for text, hidden in zip(batch, extract(batch), strict=True):
            values = hidden[:len(text)].float()
            width = min(len(text), 4)
            windows = values.unfold(0, width, 1).mean(dim=-1)
            vectors.append(torch.cat((values.mean(dim=0), windows.amax(dim=0))))
    return torch.stack(vectors)


def pair_metrics(scores, cutoff):
    predicted = [decision(row, cutoff) for row in scores]
    counts = Counter(total=len(scores))
    for row, guess in zip(scores, predicted, strict=True):
        counts.update(correct=guess == row['expected'], raw_correct=row['predicted'] == row['expected'],
                      clarify=guess == CLARIFY, wrong_action=guess != CLARIFY and guess != row['expected'])
    counts['pairs_correct'] = sum(
        left == CANCEL and right == PROCEED for left, right in zip(predicted[::2], predicted[1::2], strict=True)
    )
    return counts


@torch.inference_mode()
def choice_scores(model, tokenizer, cases):
    letters = ('A', 'B', 'C')
    labels = [tokenizer.encode(letter, add_special_tokens=False) for letter in letters]
    if any(len(label) != 1 for label in labels) or len({label[0] for label in labels}) != len(letters):
        raise ValueError('The options must be distinct single tokens')
    tokens = [label[0] for label in labels]
    scores = []
    for reply, expected in cases:
        ordered = []
        for offset in range(len(OPTIONS)):
            order = [(offset + index) % len(OPTIONS) for index in range(len(OPTIONS))]
            options = '\n'.join(f'{letter}：{OPTIONS[index]}'
                                for letter, index in zip(letters, order, strict=True))
            question = ('判断用户本人希望如何处理当前查阅。引用他人的话不算用户的决定。'
                        f'只回答 A、B 或 C。\n{options}\n用户答复：「{reply}」')
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                                                   tokenize=False, add_generation_prompt=True)
            ids = tokenizer.encode(prompt + '</think>\n\n', add_special_tokens=False)
            if len(ids) > 2048:
                raise ValueError('Choice prompt exceeded the validated position limit')
            inputs = torch.tensor([ids], device=model.device)
            logits = model(input_ids=inputs, use_cache=False, logits_to_keep=1).logits[0, -1, tokens].float()
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite choice logits')
            ordered.append(logits[torch.argsort(torch.tensor(order, device=logits.device))])
        rows = torch.stack(ordered)
        confidence, predicted = rows.mean(0).softmax(-1).max(0)
        scores.append(dict(text=reply, expected=expected, predicted=predicted.item(), confidence=confidence.item(),
                           order_predictions=rows.argmax(-1).tolist(), option_logits=rows.tolist()))
    return scores


def main(args):
    start = perf_counter()
    torch.set_num_threads(2)
    prior = json.loads((args.archive / 'result.json').read_text(encoding='utf-8'))
    archive = args.archive / 'three-versions.sqlite3'
    reader_path, normalizer, writer, binding = artifacts(args.reader, args.normalizer)
    sources = {path: base.sha(path) for path in (archive, reader_path, normalizer, writer, args.recurrent[0],
               *(directory / 'attention.pt' for directory in args.recurrent[1:]))}
    expected = {Path(path).resolve(): digest for path, digest in prior['sources'].items()}
    expected[archive.resolve()] = prior['archive_sha256']
    if any(expected.get(path.resolve()) != digest for path, digest in sources.items()):
        raise ValueError('Input weights or archive differ from the source experiment')

    train = [*EXPLICIT['train'], *DIRECT_TRAIN, *action_pairs(TRAIN),
             *((reply, CLARIFY) for reply, _ in speaker_examples(SPEAKER_TRAIN))]
    calibration = [*EXPLICIT['calibration'], *DIRECT_CALIBRATION, *action_pairs(CALIBRATION),
                   *((reply, CLARIFY) for reply, _ in speaker_examples(SPEAKER_CALIBRATION))]
    held = {family: action_pairs(rows) for family, rows in HELD.items()}
    held['unclear'] = [(reply, CLARIFY) for reply in UNCLEAR]
    splits = [{reply for reply, _ in rows} for rows in (train, calibration, *held.values())]
    if any(left & right for left, right in product(splits, splits) if left is not right):
        raise ValueError('Training and evaluation wording must be disjoint')
    values = {int(number): text for number, text in prior['values'].items()}
    if any(value in reply for split in splits for reply in split for value in values.values()):
        raise ValueError('A prompt contains a stored value')
    addresses = [tuple(MemoryAddress(*entry[index:index + 2]) for index in (0, 2)) for entry in PEOPLE['held']]

    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    results = {}
    with local_model(args.recurrent) as model, VersionStore(archive, binding) as store:
        extract = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
        weights = torch.load(reader_path, weights_only=True, map_location='cpu')
        reader = VersionReplyActions(model.config.hidden_size, weights['scaffold_tokens'], slots=2)
        reader.load_state_dict(weights, strict=True)
        reader.eval().to(model.device)
        embedding = model.get_input_embeddings().weight.detach()
        indices = compress_dimension(embedding.shape[1]).to(embedding.device)
        codebook = F.normalize(embedding.index_select(1, indices).float(), dim=-1)

        arms = (('mean', encode_texts), ('windows', window_features))
        if args.backbone_choices:
            arms = (arms[0], ('backbone_choices', None))
        for name, encode in arms:
            checkpoint_sha = None
            if encode is None:
                score = lambda cases: choice_scores(model, tokenizer, cases)
            else:
                head = fit_probe(extract, train, 3, clauses=True, encode=encode)
                checkpoint = args.output / f'{name}.pt'
                atomic_save(head.state_dict(), checkpoint)
                head.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'), strict=True)
                checkpoint_sha = base.sha(checkpoint)
                score = lambda cases: classify(head, extract, cases, clauses=True, encode=encode)
            calibrated = score(calibration)
            cutoff = calibrate(calibrated)
            outcomes = {}
            for family, cases in held.items():
                scores = score(cases)
                picks = [decision(row, cutoff) for row in scores]
                rows = [('本轮查阅该如何处理？', reply, label, family) for reply, label in cases]
                outcome = evaluate_choices(rows, picks, addresses, store, model, tokenizer, codebook, reader, values)
                outcome.update(scores=scores, metrics=pair_metrics(scores, cutoff))
                outcomes[family] = outcome
                print(name, cutoff, family, outcome['metrics'], outcome['totals'], outcome['summary'], flush=True)
            results[name] = dict(cutoff=cutoff, calibration=calibrated, outcomes=outcomes,
                                 checkpoint_sha256=checkpoint_sha)
    if any(base.sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('An input model or archive changed')
    store_json(args.output / 'result.json', dict(results=results, training=len(train),
                                               sources={str(path): digest for path, digest in sources.items()},
                                               seconds=perf_counter() - start))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('archive', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--recurrent', type=Path, nargs=3, required=True)
    cli.add_argument('--reader', type=Path, default=Path('runs/spora-dual-version-reply-v1'))
    cli.add_argument('--normalizer', type=Path, default=Path('runs/spora-version-normalizer-v1'))
    cli.add_argument('--backbone-choices', action='store_true')
    main(cli.parse_args())
