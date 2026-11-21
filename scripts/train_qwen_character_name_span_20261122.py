"""Train character-boundary name selection on frozen SPORA decoder states."""

import argparse
import json
from pathlib import Path

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import NAMES
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_full_stream_loader import load_full_streamed

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from spora.integrated_checkpoint import load_integrated
from spora.language_training import atomic_save
from spora.named_address import feature_files_sha256
from spora.research_name_span import CharacterNameSpan, MAX_NAME_LENGTH, select_character_span

INDICES = (6, 18)
SEED = 20261122
USER_PREFIX = '从提问中定位需要查询的唯一人物。若没有明确对象，或同时查询多个对象，则没有目标。\n提问：'
SINGLE = (
    '这回请核对{name}现用的{field}。', '上周填写的{name}那项{field}是哪种？',
    '你记得修改前{name}登记的是哪项{field}吗？', '我先看看{field}这一栏，查的人是{name}。',
    '麻烦打开{name}的记录，告诉我{field}是什么。', '按更正后的那张表，给{name}准备哪样{field}？',
    '想复核一下{name}写过的{field}，请说完整。', '再确认一遍，{name}用的{field}是哪款？',
)
TARGET_WITH_DISTRACTOR = (
    '先不查{other}，只告诉我{name}登记的{field}。',
    '{other}刚来问过了，现在我要查{name}原先那项{field}。',
    '{name}的{field}填什么？{other}今天只替我转达。',
    '我问的是{name}的{field}，不要跟{other}混在一起。',
)
NO_TARGET = (
    '说出{name}和{other}各自最新的{field}。',
    '先别查{name}的{field}，帮我写一句邀请。',
    '你能猜猜{name}以后会改成哪种{field}吗？',
    '这里没有人名。把{field}这个词译成英文。',
)


def examples(names, fields, *, separate_intent=False):
    rows = []
    for index, name in enumerate(names):
        other = names[(index + 3) % len(names)]
        field = fields[index % len(fields)]
        for template in (*SINGLE, *TARGET_WITH_DISTRACTOR, *NO_TARGET):
            question = template.format(name=name, other=other, field=field)
            reject = NO_TARGET if not separate_intent else (NO_TARGET[0], NO_TARGET[-1])
            rows.append((question, None if template in reject else name))
    return list(dict.fromkeys(rows))


def news_replay():
    from pyarrow import parquet
    from scripts.deepseek_cluener_locator_train import TRAIN, TRAIN_SHA, VALIDATION, VALIDATION_SHA, remove_validation_texts
    from scripts.deepseek_cluener_span_audit import labeled_names
    import hashlib

    sources = {TRAIN: TRAIN_SHA, VALIDATION: VALIDATION_SHA}
    if any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('CLUENER source digest mismatch')
    rows = remove_validation_texts(parquet.read_table(TRAIN).to_pylist(), parquet.read_table(VALIDATION).to_pylist())
    groups = {'single': [], 'multiple': [], 'none': []}
    for row in sorted(rows, key=lambda row: hashlib.sha256(row['text'].encode()).digest()):
        text = row['text']
        if not 8 <= len(text) <= 44 or any(name in text for name in NAMES):
            continue
        spans = labeled_names(row)
        if len(spans) == 1:
            start, end = spans[0]
            name = text[start:end]
            if len(name) <= MAX_NAME_LENGTH and text.count(name) == 1:
                groups['single'].append((text, name))
        else:
            groups['multiple' if spans else 'none'].append((text, None))
    replay = [row for group, limit in (('single', 128), ('multiple', 24), ('none', 24))
              for row in list(dict.fromkeys(groups[group]))[:limit]]
    return replay, sources


def splits(*, mention_replay=False):
    from scripts.audit_qwen_compatible_natural_memory_20261121 import cases, fresh_cases, fresh_version_cases

    train_names = ('小雯', '小雯鑫', '阿秋', '阿秋珂', '小宸', '小宸阳', '阿鹭', '阿鹭晨',
                   '任青禾', '江若烟', '叶佳敏', '刘平', '霍焱', '宋季夏', '蓉蓉', '周简宁',
                   '小恩', '小恩和', '阿煦', '阿煦生', '南霜', '李庭菲')
    training = examples(train_names, ('饮品', '饮料', '字号', '偏好颜色', '收件地点'),
                        separate_intent=mention_replay)
    calibration = []
    for name in ('杨沂', '阿鹏飞', '小蓓', '严岑萱'):
        calibration.extend(((f'劳驾瞧瞧{name}先前填了什么茶水。', name),
                            (f'凡是姓名为{name}的，现时所选热饮是哪款？', name)))
        if mention_replay:
            calibration.extend(((f'毋须调出{name}和唐葵的菜单，等明天再查。', None),
                                (f'{name}和尹菲各点过什么茶水？都说出来。', None)))
        else:
            calibration.extend(((f'毋须调出{name}的菜单，等明天再查。', None),
                                (f'我想跟{name}打声招呼，顺便聊聊茶水。', None)))

    def memory_examples(suite):
        return [(question, None if index is None else NAMES[index]) for question, index, _ in suite]

    challenge = []
    for index, (name, suffix) in enumerate(zip(NAMES, ('珉', '宣', '雁', '炜', '瑾', '筝', '禾', '修'), strict=True)):
        other = NAMES[(index + 3) % len(NAMES)]
        longer = name + suffix
        challenge.extend(((f'{other}不用管，我只想知道{name}的杯子里该放哪样饮品。', name),
                          (f'刚才是{other}在说话，要读取的是{name}头一回选的饮品。', name),
                          (f'把{name}的留给下回，我问的是{longer}之前常选哪款饮品。', longer),
                          (f'你把先前留给{longer}的饮品说一遍行吗？', longer),
                          (f'{name}与{other}各自那杯饮品叫什么？两杯都要。', None),
                          (f'别引述{name}写过的饮品，给{other}写一句赠言。', None)))
    return {'train': training, 'calibration': calibration, 'original': memory_examples(cases()),
            'fresh_v1': memory_examples(fresh_cases()), 'fresh_v2': memory_examples(fresh_version_cases()),
            'boundary': challenge}


def character_inputs(tokenizer, questions, features, prefix):
    """Give each character its contextual token state and position within the token."""
    width = max(map(len, questions))
    hidden = features.shape[1] * features.shape[-1]
    result = torch.zeros((len(questions), width, hidden + 4), dtype=features.dtype)
    mask = torch.arange(width)[None, :] < torch.tensor([len(q) for q in questions])[:, None]
    for i, question in enumerate(questions):
        formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': prefix + question}],
                                                  tokenize=False, add_generation_prompt=True)
        offset = formatted.find(question)
        if offset < 0 or formatted.find(question, offset + 1) >= 0:
            raise ValueError('Question must appear once in the chat template')
        pairs = tokenizer(formatted, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        # Some characters are represented by multiple byte tokens with the same offsets.
        positions = torch.arange(offset, offset + len(question))
        spans = torch.tensor(pairs)
        covers = (spans[:, 0, None] <= positions) & (spans[:, 1, None] > positions)
        token = (covers * torch.arange(1, len(spans) + 1)[:, None]).amax(0) - 1
        selected = token >= 0
        token = token.clamp_min(0)
        state = features[i].index_select(1, token).float()
        state = F.layer_norm(state, (features.shape[-1],)).transpose(0, 1).flatten(1)
        result[i, :len(question), :hidden] = state.masked_fill(~selected[:, None], 0.)
        left, right = spans[token].unbind(-1)
        length = (right - left).clamp_min(1)
        local = torch.stack(((positions - left) / length, (right - 1 - positions) / length,
                             length.clamp_max(8) / 8., selected.float()), dim=1)
        result[i, :len(question), hidden:] = local.masked_fill(~selected[:, None], 0.)
    return result, mask


def gold(question, name, size):
    if name is None:
        return size * size
    if question.count(name) != 1 or not 1 <= len(name) <= MAX_NAME_LENGTH:
        raise ValueError('Expected a unique labeled name with a supported span length')
    start = question.index(name)
    return start * size + start + len(name) - 1


def fit(features, mask, rows):
    model = CharacterNameSpan(features.shape[-1] - 4)
    target = torch.tensor([gold(q, name, features.shape[1]) for q, name in rows])
    generator = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.02)
    for epoch in range(15):
        for indices in torch.randperm(len(rows), generator=generator).split(16):
            optimizer.zero_grad(set_to_none=True)
            output = model(features[indices], mask[indices])
            loss = F.cross_entropy(output, target[indices])
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite name selection loss')
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print('epoch', epoch + 1, 'loss', float(loss.detach()), flush=True)
    return model.eval()


@torch.inference_mode()
def select(model, features, mask, margin):
    scores = torch.cat([model(x, m) for x, m in zip(features.split(16), mask.split(16), strict=True)])
    return [select_character_span(row, features.shape[1], margin) for row in scores]


def audit_margin(checkpoint, router_dir, output):
    if output.exists():
        raise FileExistsError(output)
    weights = router_dir / 'router.pt'
    metadata = json.loads((router_dir / 'result.json').read_text(encoding='utf-8'))
    digest = sha(weights)
    payload = torch.load(weights, map_location='cpu', weights_only=True)
    if (digest != metadata['router_sha256'] or payload['checkpoint_sha256'] != sha(checkpoint)
            or payload['base_sha256'] != WEIGHT_SHA or payload['feature_config_sha256'] != feature_files_sha256(MODEL)
            or tuple(payload['hidden_indices']) != INDICES or payload['user_prefix'] != USER_PREFIX):
        raise ValueError('Name head provenance mismatch')
    data = splits()
    rows = [(group, question, expected) for group in ('original', 'fresh_v1', 'fresh_v2', 'boundary')
            for question, expected in data[group]]
    questions = [question for _, question, _ in rows]
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder, _, _ = load_integrated(checkpoint, load_full_streamed(), MODEL)
    decoder.eval().requires_grad_(False)
    features, _ = extract(decoder.base, tokenizer, questions, batch_size=2, hidden_indices=INDICES,
                          user_prefix=USER_PREFIX, last_logit_only=True)
    del decoder
    features, mask = character_inputs(tokenizer, questions, features, USER_PREFIX)
    model = CharacterNameSpan(payload['feature_width']).eval()
    model.load_state_dict(payload['state'], strict=True)
    raw = select(model, features, mask, float('-inf'))
    metrics = {}
    for title, threshold in (('calibrated', payload['margin']), ('zero', 0.), ('always_select', float('-inf'))):
        metrics[title] = {}
        for (span, confidence), (group, question, expected) in zip(raw, rows, strict=True):
            selected = question[slice(*span)] if span is not None and confidence > threshold else None
            counts = metrics[title].setdefault(group, {'exact_positive': 0, 'positive': 0,
                                                        'selected_registered_wrong': 0, 'predicted': 0})
            counts['exact_positive'] += int(expected is not None and selected == expected)
            counts['positive'] += int(expected is not None)
            counts['predicted'] += int(selected is not None)
            counts['selected_registered_wrong'] += int(selected in NAMES and selected != expected)
    if sha(weights) != digest or sha(checkpoint) != payload['checkpoint_sha256']:
        raise ValueError('Input changed during name audit')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'router_sha256': digest, 'results': metrics}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metrics), flush=True)


def main(checkpoint, output, *, mention_replay=False):
    if output.exists():
        raise FileExistsError(output)
    data = splits(mention_replay=mention_replay)
    replay, sources = news_replay() if mention_replay else ([], {})
    data['train'] += replay
    boxes, all_rows = {}, []
    for name, rows in data.items():
        boxes[name] = slice(len(all_rows), len(all_rows) + len(rows))
        all_rows += rows
    questions = [question for question, _ in all_rows]
    if len(set(questions)) != len(questions):
        raise ValueError('Questions overlap across name experiment splits')
    checkpoint_sha = sha(checkpoint)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder, _, _ = load_integrated(checkpoint, load_full_streamed(), MODEL)
    decoder.eval().requires_grad_(False)
    hidden, _ = extract(decoder.base, tokenizer, questions, batch_size=2, hidden_indices=INDICES,
                        user_prefix=USER_PREFIX, last_logit_only=True)
    del decoder
    features, mask = character_inputs(tokenizer, questions, hidden, USER_PREFIX)
    del hidden
    train = boxes['train']
    model = fit(features[train], mask[train], data['train'])
    calibration = boxes['calibration']
    raw = select(model, features[calibration], mask[calibration], 0.)
    margin = max([0.] + [confidence for (span, confidence), (_, name) in zip(raw, data['calibration'], strict=True)
                         if span is not None and name is None])
    results = {}
    for name, part in boxes.items():
        spans = select(model, features[part], mask[part], margin)
        rows = []
        for (span, confidence), (question, target) in zip(spans, data[name], strict=True):
            selected = None if span is None else question[slice(*span)]
            rows.append({'question': question, 'expected': target, 'selected': selected, 'span': span,
                         'margin': confidence})
        results[name] = {'total': len(rows), 'exact': sum(row['selected'] == row['expected'] for row in rows),
                         'positive': sum(row['expected'] is not None for row in rows), 'rows': rows}
        print(name, results[name]['exact'], '/', len(rows), flush=True)
    if sha(checkpoint) != checkpoint_sha or any(sha(path) != digest for path, digest in sources.items()):
        raise ValueError('Source checkpoint changed during name training')
    output.mkdir(parents=True, exist_ok=False)
    weights = output / 'router.pt'
    atomic_save({'format': 'spora-character-name-span-v1', 'checkpoint_sha256': checkpoint_sha,
                 'base_sha256': WEIGHT_SHA, 'feature_config_sha256': feature_files_sha256(MODEL),
                 'hidden_indices': list(INDICES), 'user_prefix': USER_PREFIX, 'feature_width': features.shape[-1] - 4,
                 'margin': margin, 'state': model.state_dict()}, weights)
    (output / 'result.json').write_text(json.dumps({'router_sha256': sha(weights), 'margin': margin,
                                                   'news_replay_count': len(replay),
                                                   'sources': {str(path): digest for path, digest in sources.items()},
                                                   'splits': results}, ensure_ascii=False, indent=2) + '\n',
                                        encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--mention-replay', action='store_true')
    mode.add_argument('--audit-router', type=Path)
    args = parser.parse_args()
    if args.audit_router is None:
        main(args.checkpoint, args.output, mention_replay=args.mention_replay)
    else:
        audit_margin(args.checkpoint, args.audit_router, args.output)
