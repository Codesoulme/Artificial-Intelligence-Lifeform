"""Train a symmetric instruction ranker on frozen DeepSeek SPORA features."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_spora_dialogue_evidence import POSITIONS, generate, load_spora_for_eval, store_json
from scripts.audit_spora_revision_choices import PAIRS, content_scores, prompt_for
from scripts.deepseek_local_baseline import sha, torch
from scripts.spora_revision_rank_data import FRESH, FRESH_TERMS, training_pairs
from spora.language_training import atomic_save
from torch import nn
from torch.nn import functional as F


SEED = 20261125
EPOCHS = 18
FORMAT = 'spora-revision-ranker-v1'


def cases(pairs, view='all'):
    for _, earlier, endings, choices in pairs:
        for gold, last in enumerate(endings):
            transcript = tuple(enumerate((*earlier, last), 1))
            if view == 'last':
                transcript = transcript[-1:]
            elif view == 'empty':
                transcript = ()
            yield [(transcript, choice) for choice in choices], gold


@torch.no_grad()
def capture(chat, requests):
    tokenizer = chat.tokenizer
    model = chat.decoder.base.model
    device = chat.decoder.base.device
    unique = tuple(dict.fromkeys(requests))
    features = {}
    for start in range(0, len(unique), 10):
        batch = unique[start:start + 10]
        prompts = [prompt_for(tokenizer, transcript, f'候选安排如下：{choice}\n根据用户发言，判断它是否仍然有效。', True)
                   for transcript, choice in batch]
        tokens = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors='pt').to(device)
        attention = tokens['attention_mask']
        if attention.shape[1] > min(POSITIONS, chat.position_limit):
            raise ValueError('Ranker prompt exceeds the validated positions')
        positions = (attention * torch.arange(attention.shape[1], device=device)).amax(dim=1)
        position_ids = (attention.cumsum(dim=1) - 1).clamp_min(0)
        result = model(**tokens, position_ids=position_ids, use_cache=False, output_hidden_states=True)
        row = torch.arange(len(batch), device=device)
        values = torch.cat((result.hidden_states[12][row, positions], result.last_hidden_state[row, positions]), dim=-1)
        features.update(zip(batch, values.cpu().float(), strict=True))
        del result
        if start % 100 == 0:
            print('features', start + len(batch), '/', len(unique), flush=True)
    return features


def tensors(rows, features):
    values = torch.stack([torch.stack([features[request] for request in pair]) for pair, _ in rows])
    return values, torch.tensor([gold for _, gold in rows])


def ranker(width):
    # The same weights score both candidates without a label or absolute option position.
    return nn.Sequential(nn.LayerNorm(width, elementwise_affine=False), nn.Linear(width, 64),
                         nn.GELU(), nn.Dropout(.15), nn.Linear(64, 1, bias=False))


def fit(head, dataset):
    values, gold = dataset
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.03)
    random = torch.Generator().manual_seed(SEED)
    loss_history = []
    for _ in range(EPOCHS):
        losses = []
        for batch in torch.randperm(len(gold), generator=random).split(24):
            logits = head(values[batch]).squeeze(-1)
            loss = F.cross_entropy(logits, gold[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach().item())
        loss_history.append(sum(losses) / len(losses))
    return loss_history


@torch.no_grad()
def assess(head, dataset):
    values, gold = dataset
    logits = head(values).squeeze(-1)
    prediction = logits.argmax(dim=1)
    flipped = head(values.flip(1)).squeeze(-1).flip(1)
    if not torch.allclose(logits, flipped, rtol=1e-5, atol=1e-6):
        raise ValueError('Candidate order changed the score')
    correct = prediction == gold
    return {'correct': int(correct.sum()), 'complete_pairs': int(correct.view(-1, 2).all(dim=1).sum()),
            'picked': prediction.tolist(), 'scores': logits.tolist()}


def compare_responses(chat, prediction):
    totals = {mode: Counter() for mode in ('all_turns', 'selected', 'oracle')}
    rows = []
    cap = min(POSITIONS, chat.position_limit)
    for index, ((name, earlier, endings, choices), terms) in enumerate(zip(FRESH, FRESH_TERMS, strict=True)):
        for gold, last in enumerate(endings):
            selected = prediction[2 * index + gold]
            evidence = '\n'.join(f'第{turn}次要求：{text}' for turn, text in enumerate((*earlier, last), 1))
            prompts = {'all_turns': '下面是用户按顺序说的要求。请完成用户要求并给出可直接使用的正文。\n' + evidence,
                       'selected': '请根据以下现行安排，写一句可以直接使用的正文。\n' + choices[selected],
                       'oracle': '请根据以下现行安排，写一句可以直接使用的正文。\n' + choices[gold]}
            if name == '开门告示':
                for mode in ('selected', 'oracle'):
                    prompts[mode] = f'只输出“{choices[selected if mode == "selected" else gold]}”。'
            generated = {}
            for mode, prompt in prompts.items():
                result = (dict(generated['selected']) if mode == 'oracle' and prompt == prompts['selected'] else
                          generate(chat.decoder, chat.tokenizer, [], prompt, cap))
                checks = grade_response(result, name, choices, terms, gold)
                result.update(checks)
                totals[mode].update(checks)
                generated[mode] = result
            rows.append({'name': name, 'gold': gold, 'selected': selected, 'prompts': prompts, 'outputs': generated})
        print('responses', name, flush=True)
    return {'summary': totals, 'rows': rows}


def frozen_content_baseline(chat):
    predicted = []
    for _, earlier, endings, choices in FRESH:
        biases = content_scores(chat.decoder, chat.tokenizer, (), choices, True)
        for last in endings:
            full = content_scores(chat.decoder, chat.tokenizer, tuple(enumerate((*earlier, last), 1)), choices, True)
            predicted.append(int(full[1] - biases[1] > full[0] - biases[0]))
    correct = [value == index % 2 for index, value in enumerate(predicted)]
    return {'picked': predicted, 'correct': sum(correct),
            'complete_pairs': sum(left and right for left, right in zip(correct[::2], correct[1::2], strict=True))}


def grade_response(result, name, choices, terms, gold):
    text = result['answer']
    adopted = terms[gold] in text
    stale = terms[1 - gold] in text
    retained = all(word in text for word in terms[2])
    if name == '窗槽施工':
        retained = retained and any(phrase in text for phrase in ('留一扇窗', '窗开着', '窗户开着', '开一扇窗'))
    exact = name != '开门告示' or text.strip('。！! ') == choices[gold]
    return {'adopted': adopted, 'stale': stale, 'retained': retained, 'eos': result['eos'],
            'passed': adopted and not stale and retained and exact and result['eos']}


def summarize_saved_responses(payload):
    """Give every selector the identical short prompt and deterministic reply."""
    expected = 2 * len(FRESH)
    if len(payload['rows']) != expected or len(payload['baseline']['picked']) != expected:
        raise ValueError('Saved responses must cover both endings of every held-out scenario')
    totals = {mode: Counter() for mode in ('all_turns', 'trained', 'frozen', 'always_first', 'always_second', 'oracle')}
    rows = []
    for index, ((name, _, _, choices), terms) in enumerate(zip(FRESH, FRESH_TERMS, strict=True)):
        start = 2 * index
        pair = payload['rows'][start:start + 2]
        short = [row['outputs']['oracle'] for row in pair]
        for gold, row in enumerate(pair):
            if row['name'] != name or row['gold'] != gold:
                raise ValueError('Saved response row differs from the held-out scenario')
            picked = row['selected']
            if short[picked]['token_ids'] != row['outputs']['selected']['token_ids']:
                raise ValueError('The same short prompt returned different tokens')
            frozen = payload['baseline']['picked'][start + gold]
            outputs = {'all_turns': row['outputs']['all_turns'], 'trained': short[picked], 'frozen': short[frozen],
                       'always_first': short[0], 'always_second': short[1], 'oracle': short[gold]}
            checks = {mode: grade_response(value, name, choices, terms, gold) for mode, value in outputs.items()}
            for mode, result in checks.items():
                totals[mode].update(result)
            rows.append({'name': name, 'gold': gold, 'trained': picked, 'frozen': frozen, 'checks': checks})
    return {'summary': totals, 'rows': rows}


def main(args):
    target = args.output.resolve()
    if args.compare_saved:
        saved = target / 'responses.json'
        result = summarize_saved_responses(json.loads(saved.read_text(encoding='utf-8')))
        store_json(target / 'response-controls.json', {'input_sha256': sha(saved), **result})
        print(result['summary'], flush=True)
        return
    target.mkdir(parents=True, exist_ok=True)
    report = target / ('reload.json' if args.verify else 'result.json')
    weights = target / 'ranker.pt'
    if report.exists() or (not args.verify and weights.exists()):
        raise FileExistsError('Use a fresh experiment directory or verify the saved run')
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    train = training_pairs()
    if {row[1:] for row in train}.intersection(row[1:] for row in (*PAIRS, *FRESH)):
        raise ValueError('Evaluation pairs overlap training')
    rows = {f'{name}_{mode}': tuple(cases(pairs, mode))
            for name, pairs in (('archived', PAIRS), ('fresh', FRESH)) for mode in ('all', 'last', 'empty')}
    train_rows = tuple(cases(train))
    if not args.verify:
        rows['train_all'] = train_rows
    digest = sha256(json.dumps((train, PAIRS, FRESH, FRESH_TERMS), ensure_ascii=False).encode()).hexdigest()
    chat = load_spora_for_eval(target)
    requests = (request for group in rows.values() for pair, _ in group for request in pair)
    features = capture(chat, requests)
    width = next(iter(features.values())).numel()
    head = ranker(width)
    if args.verify:
        saved = torch.load(weights, map_location='cpu', weights_only=True)
        if (saved['format'], saved['width'], saved['data'], saved['runtime']) != (FORMAT, width, digest, chat.awake_individual):
            raise ValueError('Ranker checkpoint is incompatible with this experiment')
        head.load_state_dict(saved['weights'])
        loss = None
    else:
        loss = fit(head, tensors(train_rows, features))
        atomic_save({'format': FORMAT, 'width': width, 'data': digest, 'runtime': chat.awake_individual,
                     'weights': head.state_dict()}, weights)
    head.eval()
    metrics = {key: assess(head, tensors(value, features)) for key, value in rows.items()}
    print({key: (row['correct'], row['complete_pairs']) for key, row in metrics.items()}, flush=True)
    result = {'metrics': metrics, 'data_sha256': digest, 'weights_sha256': sha(weights), 'loss': loss,
              'training_pairs': len(train), 'updates': EPOCHS * ((len(train_rows) + 23) // 24),
              'runtime': chat.awake_individual}
    if args.verify:
        reference = json.loads((target / 'result.json').read_text(encoding='utf-8'))
        if any(value['picked'] != reference['metrics'][key]['picked'] for key, value in metrics.items()):
            raise ValueError('Reload changed the saved decisions')
    store_json(report, {**result, 'seconds': perf_counter() - started,
                        'peak_bytes': torch.cuda.max_memory_allocated()})
    if args.responses:
        baseline = frozen_content_baseline(chat)
        responses = compare_responses(chat, metrics['fresh_all']['picked'])
        store_json(target / 'responses.json', {'baseline': baseline, **responses})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--responses', action='store_true')
    parser.add_argument('--compare-saved', action='store_true')
    main(parser.parse_args())
