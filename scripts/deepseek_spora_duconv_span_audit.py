"""Read-only natural-dialogue span probe with weak knowledge-derived labels."""
import hashlib
import json
import sys
import time
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.audit_p236_duconv import ARCHIVE, ARCHIVE_SHA, conversation_key
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_spora_joint_span_pilot import JointSpan, selected_text, token_span
from scripts.deepseek_spora_prefix_features import load_prefix
from scripts.deepseek_spora_prefix_parity_audit import OUTPUT as PARITY

OUTPUT = ROOT / 'runs/deepseek-spora-duconv-span-audit-20261113.json'
PRIOR_SAMPLE = ROOT / 'runs/p236-duconv-audit-20261109/sample.jsonl'
CANDIDATES = {
    'crossed': ROOT / 'runs/deepseek-spora-joint-span-crossed-pilot-20261113',
    'diverse': ROOT / 'runs/deepseek-spora-joint-span-diverse-pilot-20261113',
}
OBJECT_PERSON = {'主演', '导演', '歌手', '作者', '作词', '作曲', '配音'}


def person_names(knowledge):
    names = set()
    if not isinstance(knowledge, list):
        return names
    for triple in knowledge:
        if not isinstance(triple, list) or len(triple) != 3 or not all(isinstance(x, str) for x in triple):
            continue
        subject, relation, obj = (x.replace(' ', '') for x in triple)
        candidate = subject if relation == '性别' and obj in ('男', '女') else obj if relation in OBJECT_PERSON else ''
        if 2 <= len(candidate) <= 4 and all('\u4e00' <= c <= '\u9fff' for c in candidate):
            names.add(candidate)
    return names


def select_candidates(lines, excluded, count=80):
    choices = []
    seen = set()
    for line in lines:
        item = json.loads(line)
        conversation = item.get('conversation')
        if not isinstance(conversation, list) or not all(isinstance(x, str) for x in conversation):
            continue
        doc = conversation_key(conversation)
        if doc in excluded or doc in seen:
            continue
        seen.add(doc)
        names = person_names(item.get('knowledge'))
        eligible = []
        for turn, raw in enumerate(conversation):
            text = raw.replace(' ', '')
            matches = [name for name in names if text.count(name) == 1]
            if 8 <= len(text) <= 100 and len(matches) == 1:
                key = hashlib.sha256((doc + '\0' + str(turn)).encode()).hexdigest()
                eligible.append({'doc': doc, 'turn': turn, 'text': text, 'name': matches[0], 'rank': key})
        if eligible:
            choices.append(min(eligible, key=lambda x: x['rank']))
    choices.sort(key=lambda x: x['rank'])
    if len(choices) < count:
        raise ValueError('Insufficient natural dialogue candidates')
    return choices[:count]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if runtime.sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Archive checksum mismatch')
    prior = {json.loads(line)['conversation_sha256'] for line in PRIOR_SAMPLE.read_text(encoding='utf-8').splitlines()}
    with ZipFile(ARCHIVE) as source, source.open('DuConv/train.txt') as stream:
        selected = select_candidates(stream, prior)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    parity = json.loads(PARITY.read_text(encoding='utf-8'))
    if (parity['source_checkpoint_sha256'] != source_sha or parity['examples'] != 10
            or parity['span_matches'] != 10 or parity['max_entity_score_difference'] > 0.001):
        raise ValueError('Prefix parity evidence missing')
    started = time.perf_counter()
    torch.set_num_threads(2)
    base, router, variant = load_prefix()
    if variant != parity['feature_variant_sha256']:
        raise ValueError('Feature variant mismatch')
    tokenizer = AutoTokenizer.from_pretrained(runtime.MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    questions = [r['text'] for r in selected]
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                           add_generation_prompt=True) for q in questions]
    encoding = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                         return_offsets_mapping=True)
    features, masks = runtime.extract(base, tokenizer, questions, batch_size=1)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1).cpu()
    heads = {}
    for label, directory in CANDIDATES.items():
        info = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        path = directory / 'span.pt'
        if runtime.sha(path) != info['weights_sha256'] or info['feature_variant_sha256'] != variant:
            raise ValueError('Candidate provenance mismatch')
        data = torch.load(path, map_location='cpu', weights_only=True)
        if data['feature_variant_sha256'] != variant:
            raise ValueError('Candidate embedded provenance mismatch')
        head = JointSpan(vectors.shape[-1], width=data['width']).eval()
        head.load_state_dict(data['state'], strict=True)
        heads[label] = head
    old = nn.Linear(vectors.shape[-1], 1).eval()
    old.load_state_dict(router['locator_state'], strict=True)
    rows = []
    with torch.no_grad():
        for i, row in enumerate(selected):
            start = texts[i].find(row['text'])
            labels = token_span(encoding['offset_mapping'][i], masks[i], start, row['text'], row['name'])
            predictions = {}
            for label, head in heads.items():
                index = int(head(vectors[i], masks[i]).argmax())
                predictions[label] = selected_text(index, encoding['offset_mapping'][i], start, row['text'])
            chosen, single = single_span_mask(old(vectors[i]).squeeze(-1).sigmoid(), masks[i],
                                              router['locator_threshold'])
            positions = chosen.nonzero(as_tuple=True)[0].tolist()
            predictions['default'] = (selected_text(1 + positions[0] * len(masks[i]) + positions[-1],
                                      encoding['offset_mapping'][i], start, row['text'])
                                      if single and positions else None)
            rows.append({'doc_sha256': row['doc'], 'turn': row['turn'],
                         'text_sha256': hashlib.sha256(row['text'].encode('utf-8')).hexdigest(),
                         'target_sha256': hashlib.sha256(row['name'].encode('utf-8')).hexdigest(),
                         'representable': labels is not None,
                         'predictions': {k: {'matches_weak_label': v == row['name'], 'abstained': v is None}
                                         for k, v in predictions.items()}})
    if runtime.sha(ARCHIVE) != ARCHIVE_SHA or runtime.sha(runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected input changed')
    summary = {'total': len(rows), 'representable': sum(r['representable'] for r in rows)}
    for label in ('default', 'crossed', 'diverse'):
        summary[label] = {metric: sum(r['predictions'][label][metric] for r in rows)
                          for metric in ('matches_weak_label', 'abstained')}
    result = {'status': 'weak_label_development_probe_no_deployment', 'summary': summary, 'rows': rows,
              'archive_sha256': ARCHIVE_SHA, 'checkpoint_sha256': source_sha,
              'candidate_sha256': {label: runtime.sha(directory / 'span.pt') for label, directory in CANDIDATES.items()},
              'seconds': time.perf_counter() - started,
              'limitations': ['Names inferred from knowledge triples, not gold span annotations.',
                              'A matching person name need not be the target of the utterance or a memory request.',
                              'The knowledge list may omit other people; no no-person or multiple-person gold labels.',
                              'Chinese word segmentation spaces were removed, otherwise original single turns used.',
                              'DuConv train partition was previously examined; base pretraining overlap unknown.',
                              'Dataset rights beyond the card declaration remain unresolved; raw text is not exported.',
                              'Prefix/full-model parity checked on ten other cases only; no real memory reads or outputs.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
