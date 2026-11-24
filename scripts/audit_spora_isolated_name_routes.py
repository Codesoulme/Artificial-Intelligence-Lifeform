"""Resolve a person mention before activating a saved low-rank memory bank."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_isolated_fast_weights import NAMES, SOURCE, base, pilot, restore
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.explicit_span_address import explicit_span_address
from spora.identity_fingerprints import IdentityFingerprints
from spora.language_training import atomic_save
from spora.research_ttt_ntp import attach_read_adapters, install


SEED = 26301
TRAIN_NAMES = ('洛依', '钱翊涵', '孟奚', '沈柯明', '范霖', '罗絮文', '何雨晴', '宿宛', '许冉青', '徐春泽',
               '梁白', '夏栩', '章亦成', '周子槐', '魏雁', '江瑶', '小南', '冯茜', '何映阳', '苏扬')
CAL_NAMES = ('付恬', '于默欣', '高瑾', '俞之恒', '聂蓉', '余新然', '路琳', '成凯涛')
TRAIN = ('{a}平常喜欢喝什么饮料？', '翻翻记录，{a}爱喝哪一种？', '你以前记过{a}爱喝什么吗？',
         '说说{a}的饮品喜好。', '{b}先跳过，{a}喜欢喝的东西是什么？',
         '问的是{a}平常喝哪种饮料，先不问{b}。', '你说的是{b}吧，我查的是{a}的饮品偏好。')
CAL = ('查一下{a}最爱选的饮品。', '今天先问{a}最常点什么，暂时略过{b}。',
       '不用讲{b}，帮我核对{a}以前说爱喝什么。')
HELD = ('说起{a}，有什么饮品是常点的吗？', '饮料这块，你还记得{a}偏爱哪样不？',
        '先搁下{b}的口味，{a}平时最愿意喝哪种？', '我关心{a}那份饮料记录，{b}的先放一边。')
NEUTRAL = {
    'train': ('给{a}和{b}各推荐一款夏天喝的东西。', '{a}与{b}爱喝什么，各说一种。', '饮料要放冰箱吗？',
              '{a}咖啡馆几点开门？', '{a}这次说窗户漏水，应该找谁修理？'),
    'cal': ('{a}、{b}分别偏好哪种饮料？', '{a}酒楼周末还营业吗？', '刚泡好的茶怎样尽快降温？'),
    'held': ('把{a}还有{b}的常喝饮品分别告诉我。', '{a}茶室门口有停车位吗？', '家里没有冰块，怎么让饮料凉一点？'),
}


class MentionHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(hidden, 64), nn.GELU())
        self.endpoints = nn.Linear(64, 2)
        self.width_bias = nn.Parameter(torch.zeros(3))
        self.none = nn.Linear(64, 1)

    def forward(self, features, valid):
        features = self.project(F.normalize(features.float(), dim=-1))
        first, last = self.endpoints(features).unbind(-1)
        order = torch.arange(valid.shape[1])
        distance = order[None, :] - order[:, None]
        legal = (distance >= 1) & (distance <= 3)
        logits = first[:, :, None] + last[:, None, :] + self.width_bias[(distance - 1).clamp(0, 2)]
        logits = logits.masked_fill(~(legal & valid[:, :, None] & valid[:, None, :]), -torch.inf)
        pool = features.masked_fill(~valid[..., None], 0.).sum(1) / valid.sum(-1, keepdim=True)
        return torch.cat((self.none(pool), logits.flatten(1)), dim=1)


def authored(names, forms, neutral):
    rows = []
    for index, name in enumerate(names):
        other = names[(index + 1) % len(names)]
        for form in forms:
            rows.append({'text': form.format(a=name, b=other), 'target': name, 'kind': 'memory'})
        for form in neutral:
            rows.append({'text': form.format(a=name, b=other), 'target': None, 'kind': 'other'})
    return list({row['text']: row for row in rows}.values())


def evaluation():
    rows = []
    for i, name in enumerate(NAMES):
        other = NAMES[i ^ 1]
        rows.append({'text': HELD[i % 4].format(a=name, b=other), 'target': name, 'kind': 'known'})
        unfamiliar = name + ('明' if i % 2 else '卓')
        rows.append({'text': HELD[i % 4].format(a=unfamiliar, b=other), 'target': unfamiliar, 'kind': 'longer_name'})
    for i, name in enumerate(('小召', '周宇', '林笑', '邓新', '刘晴芸', '顾岸舟', '任诗羽', '段园')):
        rows.append({'text': HELD[i % 4].format(a=name, b=NAMES[i]), 'target': name, 'kind': 'other_unknown'})
    for i, name in enumerate(NAMES[:12]):
        form = NEUTRAL['held'][i % 3]
        rows.append({'text': form.format(a=name, b=NAMES[i ^ 1]), 'target': None,
                     'kind': 'ambiguous' if i % 3 == 0 else 'ordinary'})
    return list({row['text']: row for row in rows}.values())


def encode(extractor, rows):
    texts = [row['text'] for row in rows]
    width = max(map(len, texts))
    parts = [F.pad(extractor(texts[i:i + 32]), (0, 0, 0, width - max(map(len, texts[i:i + 32]))))
             for i in range(0, len(texts), 32)]
    valid = torch.arange(width)[None, :] < torch.tensor(list(map(len, texts)))[:, None]
    targets = []
    for row in rows:
        text, name = row['text'], row['target']
        if name is None:
            targets.append(0)
        elif text.count(name) != 1 or not 2 <= len(name) <= 4:
            raise ValueError('Expected one bounded target span')
        else:
            start = text.index(name)
            targets.append(1 + start * width + start + len(name) - 1)
    return torch.cat(parts), valid, torch.tensor(targets)


def fit(head, data):
    features, valid, targets = data
    opt = torch.optim.AdamW(head.parameters(), lr=.002)
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(20):
        for batch in torch.randperm(len(features), generator=generator).split(32):
            loss = F.cross_entropy(head(features[batch], valid[batch]), targets[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch in (4, 9, 19):
            print('locator', epoch + 1, float(loss.detach()), flush=True)


@torch.inference_mode()
def locate(head, data, rows):
    features, valid, _ = data
    logits = head(features, valid)
    best, index = logits[:, 1:].max(1)
    width = features.shape[1]
    results = []
    for row, flat, margin in zip(rows, index.tolist(), (best - logits[:, 0]).tolist(), strict=True):
        start, end = divmod(flat, width)
        results.append({'span': [start, end + 1], 'name': row['text'][start:end + 1], 'margin': margin})
    return results


def threshold(rows, picks):
    wrong = [prediction['margin'] for row, prediction in zip(rows, picks, strict=True)
             if prediction['name'] != row['target']]
    return max([0., *wrong])


def route(bank, question, found, cutoff):
    if found['margin'] <= cutoff:
        return None
    return explicit_span_address(bank, question, found['span'], found['name'])['key']


def exact(result, value):
    return result is not None and result['answer'].strip().rstrip('。.!！ ').strip() == value


def canonical_probe(model, tokenizer, memories, rows, prior):
    reads = []
    for row in rows:
        chosen = row['address']
        if chosen is None:
            continue
        response = pilot.generate(model, tokenizer, f'请只回答{chosen}喜欢喝什么饮料。', memories[chosen])
        if response != prior[chosen]['answers']['isolated']:
            raise ValueError('Canonical question differs from the unchanged isolated run')
        reads.append({'text': row['text'], 'kind': row['kind'], 'selected': chosen, 'answer': response})
    known = {row['text']: row['target'] for row in rows if row['kind'] == 'known'}
    return {'known_turns': len(known),
            'known_exact': sum(row['text'] in known and row['selected'] == known[row['text']]
                               and exact(row['answer'], prior[row['selected']]['expected']) for row in reads),
            'unknown_reads': sum(row['kind'] in ('longer_name', 'other_unknown') for row in reads), 'rows': reads}


def main(args):
    start = perf_counter()
    torch.set_num_threads(2)
    out = args.output
    if not args.verify:
        out.mkdir(parents=True, exist_ok=False)
    files = (base.MODEL / 'model.safetensors', SOURCE / 'adapters.pt', args.saved / 'result.json',
             args.saved / '32-isolated.pt')
    hashes = {str(path): base.sha(path) for path in files}
    parent = json.loads(files[2].read_text(encoding='utf-8'))
    saved_stage = parent['stages'][-1]
    if (saved_stage['count'] != 32 or hashes[str(files[3])] != saved_stage['storage']['isolated']['sha256']
            or any(hashes[str(path)] != parent['sources'][str(path)] for path in files[:2])):
        raise ValueError('Saved isolated memory binding changed')
    expected = json.loads((out / 'result.json').read_text(encoding='utf-8')) if args.verify else None
    weights = out / 'locator.pt'
    if expected is not None and (expected['sources'] != hashes or expected['locator_sha256'] != base.sha(weights)):
        raise ValueError('Locator replay sources changed')
    adapter = torch.load(files[1], weights_only=True, map_location='cpu')
    model = load_full_streamed().eval()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    wrappers = install(model, tuple(adapter['layers']))
    for layer, reader in attach_read_adapters(wrappers, rank=adapter['rank']).items():
        reader.load_state_dict(adapter['adapters'][str(layer)], strict=True)
    model.requires_grad_(False)
    memory = restore(torch.load(files[3], map_location='cpu', weights_only=True), wrappers)
    if set(memory) != set(NAMES):
        raise ValueError('Enrollment does not match the saved cases')
    enrollment = IdentityFingerprints(len(NAMES))
    for key, name in enumerate(NAMES):
        enrollment.bind(key, name)
    extractor = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=12)
    torch.manual_seed(SEED)
    head = MentionHead(model.config.hidden_size)
    if args.verify:
        head.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    else:
        fit(head, encode(extractor, authored(TRAIN_NAMES, TRAIN, NEUTRAL['train'])))
        atomic_save(head.state_dict(), weights)
    head.eval()
    cal = authored(CAL_NAMES, CAL, NEUTRAL['cal'])
    cutoff = threshold(cal, locate(head, encode(extractor, cal), cal))
    facts = {row['name']: row['expected'] for row in saved_stage['rows']}
    rows = evaluation()
    decisions = locate(head, encode(extractor, rows), rows)
    results = []
    for row, found in zip(rows, decisions, strict=True):
        key = route(enrollment, row['text'], found, cutoff)
        chosen = NAMES[key] if key is not None else None
        matches = enrollment.matching_spans(row['text'], max_name_chars=4)
        matched_keys = {key for _, _, key in matches}
        surface = next(iter(matched_keys)) if len(matched_keys) == 1 else None
        target = row['target']
        proposed = pilot.generate(model, tokenizer, row['text'], memory[chosen]) if chosen is not None else None
        control = (proposed if chosen == target else pilot.generate(model, tokenizer, row['text'], memory[target])) if target in memory else None
        results.append({**row, 'location': found, 'address': chosen, 'substring_address': NAMES[surface] if surface is not None else None,
                        'answer': proposed, 'given_address': control,
                        'memory_reply': proposed['answer'] if proposed is not None else '没有找到可读取的饮品记录。'})
    if any(base.sha(path) != digest for path, digest in zip(files, hashes.values(), strict=True)):
        raise ValueError('A frozen input changed')
    summaries = {}
    for kind in dict.fromkeys(row['kind'] for row in results):
        selected = [row for row in results if row['kind'] == kind]
        target_for = lambda row: row['target'] if row['target'] in memory else None
        summaries[kind] = {'count': len(selected), 'span_exact': sum(row['location']['name'] == row['target'] for row in selected),
                           'correct_route': sum(row['address'] == target_for(row) for row in selected),
                           'substring_correct': sum(row['substring_address'] == target_for(row) for row in selected),
                           'activated': sum(row['address'] is not None for row in selected),
                           'answer_exact': sum(exact(row['answer'], facts.get(row['target'])) for row in selected),
                           'given_address_exact': sum(exact(row['given_address'], facts.get(row['target'])) for row in selected)}
    print(summaries, flush=True)
    report = {'sources': hashes, 'locator_sha256': base.sha(weights), 'cutoff': cutoff, 'summary': summaries, 'rows': results}
    if expected is not None and any(expected[key] != value for key, value in report.items()):
        raise ValueError('Independent reload changed routing or output')
    if args.verify:
        canonical = canonical_probe(model, tokenizer, memory, results, {row['name']: row for row in saved_stage['rows']})
        store_json(out / 'canonical.json', canonical)
        print('canonical', {key: n for key, n in canonical.items() if key != 'rows'}, flush=True)
    store_json(out / ('reload.json' if args.verify else 'result.json'), {**report, 'seconds': perf_counter() - start})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('saved', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
