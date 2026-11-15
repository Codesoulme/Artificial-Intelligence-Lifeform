"""Select held-out dialogue responses for independent development review.

Dataset-derived person hints are not gold labels. Raw source text is printed
only for review and is not saved in the repository.
"""
import json
import sys
import time
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_duconv_dev_identity_audit as dev
from scripts import deepseek_spora_duconv_span_audit as natural
from scripts import deepseek_spora_natural_span_adaptation as adaptation
from scripts import deepseek_spora_name_extraction_audit as synthetic
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.audit_p236_duconv import ARCHIVE, ARCHIVE_SHA, conversation_key

COUNT = 12
MANIFEST_SHA = '023023ed147c9f06686cc92c78e94bfe3ab17c9ee8a37de2b944d269ce755a55'
OUTPUT = ROOT / 'runs/deepseek-spora-natural-span-test-audit-20261116.json'


def select_sample():
    if runtime.sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Source archive mismatch')
    reviewed = dev.reviewed_sample(dev.select_sample())
    with ZipFile(ARCHIVE) as archive:
        train_lines = list(archive.open('DuConv/train.txt'))
        test_lines = list(archive.open('DuConv/test_1.txt'))
    training = adaptation.build_training(train_lines, reviewed)
    prohibited_names = {row['subject'] for row in training if row['subject']}
    prohibited_names |= {name for row in reviewed for name in row['names']}
    prohibited_names |= {row['subject'] for row in reviewed if row['subject']}
    prohibited_names |= set(synthetic.ENROLLED)
    train_texts = {text.replace(' ', '') for item in (json.loads(line) for line in train_lines)
                   for text in item.get('conversation', []) if isinstance(text, str)}
    train_texts |= {row['text'] for row in reviewed}
    groups = {'single': [], 'none': [], 'multiple': []}
    for line in test_lines:
        item = json.loads(line)
        history, response = item.get('history'), item.get('response')
        if not isinstance(history, list) or not all(isinstance(x, str) for x in history) or not isinstance(response, str):
            continue
        text = response.replace(' ', '')
        if not 8 <= len(text) <= 100 or text in train_texts or any(name in text for name in prohibited_names):
            continue
        names = sorted(name for name in natural.person_names(item.get('knowledge')) if text.count(name) == 1)
        group = 'none' if not names else 'single' if len(names) == 1 else 'multiple'
        doc = conversation_key(history + [response])
        groups[group].append({'text': text, 'names': names, 'group': group,
                              'doc': doc, 'rank': dev.digest(doc + '\0' + text)})
    selected, used_docs, used_text = [], set(), set()
    for group in ('single', 'none', 'multiple'):
        before = len(selected)
        for item in sorted(groups[group], key=lambda row: row['rank']):
            if item['doc'] in used_docs or item['text'] in used_text:
                continue
            selected.append(item)
            used_docs.add(item['doc'])
            used_text.add(item['text'])
            if len(selected) - before == COUNT:
                break
        if len(selected) - before != COUNT:
            raise ValueError('Insufficient distinct test responses')
    return selected


def preview():
    rows = select_sample()
    print(json.dumps([{'index': index, 'group': row['group'], 'text': row['text'],
                       'names': row['names'], 'text_sha256': dev.digest(row['text'])}
                      for index, row in enumerate(rows)], ensure_ascii=True))


def reviewed_sample(rows):
    if len(rows) != 36 or dev.digest(''.join(dev.digest(row['text']) for row in rows)) != MANIFEST_SHA:
        raise ValueError('Reviewed test sample changed')
    result = []
    for index, row in enumerate(rows):
        if index in (4, 9):
            group, subject = 'multiple', None
        elif index == 15:
            group, subject = 'single', row['text'][11:17]
            if row['text'][10:11] != '是' or row['text'][17:18] != '。':
                raise ValueError('Reviewed foreign-name boundaries changed')
        elif index < 12:
            group, subject = 'single', row['names'][0]
        elif index < 24:
            group, subject = 'none', None
        else:
            group, subject = 'multiple', None
        result.append({**row, 'reviewed_group': group, 'subject': subject})
    return result


def audit():
    import torch
    from torch.nn import functional as F
    from transformers import AutoTokenizer
    from scripts.deepseek_spora_joint_span_pilot import JointSpan, selected_text, token_span
    from scripts.deepseek_spora_prefix_features import load_prefix
    from scripts.deepseek_spora_prefix_parity_audit import OUTPUT as PARITY

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    selected = reviewed_sample(select_sample())
    started = time.perf_counter()
    source_sha = runtime.sha(runtime.CHECKPOINT)
    parity = json.loads(PARITY.read_text(encoding='utf-8'))
    if (parity['source_checkpoint_sha256'] != source_sha or parity['examples'] != 10
            or parity['span_matches'] != 10 or parity['max_entity_score_difference'] > .001):
        raise ValueError('Prefix parity evidence missing')
    torch.set_num_threads(2)
    model, _, variant = load_prefix()
    if variant != parity['feature_variant_sha256']:
        raise ValueError('Feature variant mismatch')
    tokenizer = AutoTokenizer.from_pretrained(runtime.MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    questions = [row['text'] for row in selected]
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                           add_generation_prompt=True) for question in questions]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                        return_offsets_mapping=True)
    features, masks = runtime.extract(model, tokenizer, questions, batch_size=1)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],))
                         for i in range(2)], dim=-1).cpu()
    masks = masks.cpu()
    directories = {'baseline': natural.CANDIDATES['crossed'],
                   'balanced': adaptation.OUT,
                   'more_multiple': ROOT / 'runs/deepseek-spora-natural-span-more-multiple-20261116'}
    heads, hashes = {}, {}
    for label, directory in directories.items():
        report = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        path = directory / 'span.pt'
        hashes[label] = runtime.sha(path)
        expected = report['weights_sha256'] if label == 'baseline' else report['candidate_sha256']
        if hashes[label] != expected or report['feature_variant_sha256'] != variant:
            raise ValueError('Candidate digest or feature mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['feature_variant_sha256'] != variant:
            raise ValueError('Embedded feature mismatch')
        head = JointSpan(vectors.shape[-1], width=payload['width']).eval()
        head.load_state_dict(payload['state'], strict=True)
        heads[label] = head
    rows = []
    with torch.no_grad():
        for index, row in enumerate(selected):
            start = texts[index].find(row['text'])
            gold = token_span(encoded['offset_mapping'][index], masks[index], start, row['text'], row['subject'])
            predictions = {}
            for label, head in heads.items():
                choice = int(head(vectors[index], masks[index]).argmax())
                answer = selected_text(choice, encoded['offset_mapping'][index], start, row['text'])
                predictions[label] = {'correct': answer == row['subject'], 'selected': answer is not None}
            rows.append({'doc_sha256': row['doc'], 'text_sha256': dev.digest(row['text']),
                         'group': row['reviewed_group'], 'representable': gold is not None,
                         'predictions': predictions})
    if runtime.sha(ARCHIVE) != ARCHIVE_SHA or runtime.sha(runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected source changed')
    summary = {}
    for group in ('single', 'none', 'multiple'):
        items = [row for row in rows if row['group'] == group]
        summary[group] = {'total': len(items), 'representable': sum(row['representable'] for row in items),
                          **{name: {metric: sum(row['predictions'][name][metric] for row in items)
                                    for metric in ('correct', 'selected')}
                             for name in heads}}
    result = {'status': 'agent_reviewed_test_partition_development_only', 'summary': summary, 'rows': rows,
              'sample_manifest_sha256': MANIFEST_SHA, 'archive_sha256': ARCHIVE_SHA,
              'source_checkpoint_sha256': source_sha, 'feature_variant_sha256': variant,
              'candidate_sha256': hashes, 'seconds': time.perf_counter() - started,
              'annotation': 'Codex reviewed all 36 single-turn responses before model evaluation; no second review.',
              'limitations': ['Weak knowledge hints and single-agent annotation; selected sample not representative.',
                              'Exact train/development text and selected training names excluded; pretraining overlap unknown.',
                              'No multi-turn context, authorization, actual memory read or natural memory updates.',
                              'No-person and multi-person abstention are experimental policies, not universal requirements.',
                              'Crossed and both natural heads use an older frozen SPORA-graft prefix.',
                              'Only ten other examples established prefix/full-model parity.',
                              'The more-multiple training schedule was chosen after looking at the development sample.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--preview']:
        preview()
    elif not sys.argv[1:]:
        audit()
    else:
        raise SystemExit('Usage: script [--preview]')
