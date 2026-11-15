"""Select a disjoint DuConv development sample for manual span review.

Only digests and metrics are persisted. Knowledge triples are selection hints,
not gold labels; the audit requires separately reviewed annotations.
"""
import hashlib
import json
import sys
import time
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_local_baseline as base
from scripts.audit_p236_duconv import ARCHIVE, ARCHIVE_SHA, conversation_key
from scripts.deepseek_spora_duconv_span_audit import person_names, select_candidates

SINGLE_COUNT = 24
OTHER_COUNT = 12
MANIFEST_SHA = 'd09d6767a0915d941339b977ca7483165f8fd560d3bdc2ab746f7a6fc423d7a3'
OUTPUT = ROOT / 'runs/deepseek-spora-duconv-dev-identity-audit-20261116.json'
GENERATION_OUTPUT = ROOT / 'runs/deepseek-spora-duconv-dev-generation-audit-20261116.json'


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def select_sample():
    if base.sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Source archive checksum mismatch')
    with ZipFile(ARCHIVE) as source:
        train_docs = {conversation_key(row['conversation']) for row in
                      (json.loads(line) for line in source.open('DuConv/train.txt'))
                      if isinstance(row.get('conversation'), list)
                      and all(isinstance(x, str) for x in row['conversation'])}
        dev_lines = list(source.open('DuConv/dev.txt'))
    singles = select_candidates(dev_lines, train_docs, SINGLE_COUNT)
    used_docs = {row['doc'] for row in singles}
    groups = {'none': [], 'multi': []}
    for line in dev_lines:
        item = json.loads(line)
        conversation = item.get('conversation')
        if not isinstance(conversation, list) or not all(isinstance(x, str) for x in conversation):
            continue
        doc = conversation_key(conversation)
        if doc in train_docs or doc in used_docs:
            continue
        names = person_names(item.get('knowledge'))
        for turn, raw in enumerate(conversation):
            question = raw.replace(' ', '')
            mentions = sorted(name for name in names if question.count(name) == 1)
            group = 'none' if not mentions else 'multi' if len(mentions) >= 2 else None
            if group and 8 <= len(question) <= 100:
                rank = digest(doc + '\0' + str(turn))
                groups[group].append({'doc': doc, 'turn': turn, 'text': question,
                                      'names': mentions, 'rank': rank})
    selected = [{'group': 'single', **row, 'names': [row['name']]} for row in singles]
    seen_text = {row['text'] for row in selected}
    for group in ('none', 'multi'):
        count = 0
        for row in sorted(groups[group], key=lambda item: item['rank']):
            if row['doc'] in used_docs or row['text'] in seen_text:
                continue
            selected.append({'group': group, **row})
            used_docs.add(row['doc'])
            seen_text.add(row['text'])
            count += 1
            if count == OTHER_COUNT:
                break
        if count != OTHER_COUNT:
            raise ValueError('Insufficient distinct dialogue groups')
    if len({row['doc'] for row in selected}) != len(selected):
        raise ValueError('Repeated conversation')
    return selected


def preview():
    rows = select_sample()
    print(json.dumps([{'index': i, 'group': row['group'], 'text': row['text'],
                       'names': row['names'], 'text_sha256': digest(row['text'])}
                      for i, row in enumerate(rows)], ensure_ascii=True))


def reviewed_sample(rows):
    if len(rows) != 48 or digest(''.join(digest(row['text']) for row in rows)) != MANIFEST_SHA:
        raise ValueError('Reviewed sample changed')
    reviewed = []
    for index, row in enumerate(rows):
        if index in (25, 30):
            continue
        if index == 2:
            group, subject = 'multiple', None
        elif index == 24:
            group, subject = 'single', row['text'][:3]
            if row['text'][3:4] != '，':
                raise ValueError('Reviewed foreign name boundary changed')
        elif index < 24:
            group, subject = 'single', row['names'][0]
        elif index < 36:
            group, subject = 'none', None
        else:
            group, subject = 'multiple', None
        reviewed.append({**row, 'reviewed_group': group, 'subject': subject})
    return reviewed


def audit():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from transformers import AutoTokenizer
    from scripts import deepseek_spora_runtime_chat as runtime
    from scripts.deepseek_learned_name_span import single_span_mask
    from scripts.deepseek_spora_duconv_span_audit import CANDIDATES
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
    model, router, variant = load_prefix()
    if variant != parity['feature_variant_sha256']:
        raise ValueError('Feature variant mismatch')
    tokenizer = AutoTokenizer.from_pretrained(runtime.MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    questions = [row['text'] for row in selected]
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                           add_generation_prompt=True) for q in questions]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                        return_offsets_mapping=True)
    features, masks = runtime.extract(model, tokenizer, questions, batch_size=1)
    vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],))
                         for i in range(2)], dim=-1).cpu()
    masks = masks.cpu()
    heads, candidate_hashes = {}, {}
    for label, directory in CANDIDATES.items():
        info = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        path = directory / 'span.pt'
        candidate_hashes[label] = runtime.sha(path)
        if candidate_hashes[label] != info['weights_sha256'] or info['feature_variant_sha256'] != variant:
            raise ValueError('Candidate provenance mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['feature_variant_sha256'] != variant:
            raise ValueError('Candidate embedded feature variant mismatch')
        head = JointSpan(vectors.shape[-1], width=payload['width']).eval()
        head.load_state_dict(payload['state'], strict=True)
        heads[label] = head
    default = nn.Linear(vectors.shape[-1], 1).eval()
    default.load_state_dict(router['locator_state'], strict=True)
    rows = []
    with torch.no_grad():
        for index, row in enumerate(selected):
            question = row['text']
            start = texts[index].find(question)
            gold = token_span(encoded['offset_mapping'][index], masks[index], start, question, row['subject'])
            predictions = {}
            for label, head in heads.items():
                choice = int(head(vectors[index], masks[index]).argmax())
                predictions[label] = selected_text(choice, encoded['offset_mapping'][index], start, question)
            chosen, single = single_span_mask(default(vectors[index]).squeeze(-1).sigmoid(), masks[index],
                                              router['locator_threshold'])
            positions = chosen.nonzero(as_tuple=True)[0].tolist()
            predictions['default'] = (selected_text(1 + positions[0] * len(masks[index]) + positions[-1],
                                                    encoded['offset_mapping'][index], start, question)
                                      if single and positions else None)
            rows.append({'doc_sha256': row['doc'], 'turn': row['turn'], 'text_sha256': digest(question),
                         'group': row['reviewed_group'], 'representable': gold is not None,
                         'predictions': {key: {'correct': value == row['subject'],
                                               'selected': value is not None}
                                         for key, value in predictions.items()}})
    if runtime.sha(ARCHIVE) != ARCHIVE_SHA or runtime.sha(runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected source changed')
    summary = {}
    for group in ('single', 'none', 'multiple'):
        items = [row for row in rows if row['group'] == group]
        summary[group] = {'total': len(items), 'representable': sum(row['representable'] for row in items),
                          **{name: {metric: sum(row['predictions'][name][metric] for row in items)
                                    for metric in ('correct', 'selected')}
                             for name in ('default', 'crossed', 'diverse')}}
    report = {'status': 'agent_reviewed_development_sample_no_deployment', 'summary': summary, 'rows': rows,
              'sample_manifest_sha256': MANIFEST_SHA, 'archive_sha256': ARCHIVE_SHA,
              'source_checkpoint_sha256': source_sha, 'feature_variant_sha256': variant,
              'candidate_sha256': candidate_hashes, 'seconds': time.perf_counter() - started,
              'annotation': 'Codex reviewed 48 single turns; two ambiguous examples excluded; no human second review.',
              'limitations': ['Original public dialogue, but selected using knowledge triples and limited to one turn.',
                              'Subject labels are agent-reviewed named mentions, not memory-read intent or identity.',
                              'No-name and multi-name abstention is a diagnostic policy, not a universal task requirement.',
                              'Dev conversations were excluded if byte-identical to train; person names may overlap.',
                              'No guarantee of base pretraining independence or data rights beyond declared card.',
                              'Prefix/full-model parity checked on only ten other examples.',
                              'No real individual read, write, answer generation, or multi-turn context.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def audit_generation():
    import torch
    from scripts import deepseek_spora_adapter_restart_audit as binding_source
    from scripts import deepseek_spora_name_extraction_audit as extraction
    from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
    from scripts.deepseek_spora_full_stream_loader import load_full_streamed

    if GENERATION_OUTPUT.exists():
        raise FileExistsError(GENERATION_OUTPUT)
    selected = reviewed_sample(select_sample())
    started = time.perf_counter()
    base_sha = base.sha(base.MODEL / 'model.safetensors')
    feature_sha, files = binding_source.binding()
    if base_sha != base.WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    torch.manual_seed(17381)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    rows = []
    for index, row in enumerate(selected):
        inputs = pilot.closed_think_inputs(tokenizer, extraction.INSTRUCTION + row['text'], 'cuda')
        with torch.inference_mode():
            output = model.generate(**inputs, use_cache=True, max_new_tokens=48, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        token_ids = output[0, inputs['input_ids'].shape[1]:].tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        strict, strict_status = extraction.parse_response(text, row['text'])
        embedded, embedded_status = extraction.parse_embedded_response(text, row['text'])
        rows.append({'doc_sha256': row['doc'], 'turn': row['turn'], 'text_sha256': digest(row['text']),
                     'output_sha256': digest(text), 'group': row['reviewed_group'],
                     'strict': {'correct': strict == row['subject'] and strict_status in ('name', 'null'),
                                'selected': strict is not None, 'status': strict_status},
                     'embedded': {'correct': embedded == row['subject'] and embedded_status in ('name', 'null'),
                                  'selected': embedded is not None, 'status': embedded_status}})
        if (index + 1) % 10 == 0:
            print(json.dumps({'generated': index + 1, 'total': len(selected)}), flush=True)
    if base.sha(base.MODEL / 'model.safetensors') != base_sha or base.sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError('Protected source changed')
    summary = {}
    for group in ('single', 'none', 'multiple'):
        items = [row for row in rows if row['group'] == group]
        summary[group] = {'total': len(items), **{name: {metric: sum(row[name][metric] for row in items)
                         for metric in ('correct', 'selected')} for name in ('strict', 'embedded')}}
    report = {'status': 'agent_reviewed_generation_development_no_deployment', 'summary': summary,
              'rows': rows, 'sample_manifest_sha256': MANIFEST_SHA, 'archive_sha256': ARCHIVE_SHA,
              'base_sha256': base_sha, 'feature_sha256': feature_sha, 'feature_files': files,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Prompted extraction on frozen base, not the SPORA graft used by the locator audit.',
                              'Strict and embedded parsers are diagnostics, not identity security boundaries.',
                              'Annotation reviewed by one agent; no independent human review.',
                              'Single turns selected using incomplete knowledge hints; no actual memory intent labels.',
                              'No-person and multi-person abstention is a diagnostic rule, not universal task behavior.',
                              'Greedy generation, closed think prefix, and 48-token cap.',
                              'No read, write, deployed routing or multi-turn context tested.']}
    with GENERATION_OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--preview']:
        preview()
    elif sys.argv[1:] == ['--generate']:
        audit_generation()
    elif not sys.argv[1:]:
        audit()
    else:
        raise SystemExit('Usage: script [--preview|--generate]')
