"""Diagnose whether existing token masks can represent gold name boundaries."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import pyarrow.parquet as parquet
import torch
from transformers import AutoTokenizer

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_cluener_span_audit import SOURCE, SOURCE_SHA, character_mask
from scripts.deepseek_cluener_locator_train import TRAIN, TRAIN_SHA
from scripts.deepseek_pooled_address import question_mask
from scripts.deepseek_spora_conversation_locator_audit import OUTPUT as INPUT, select_names

OUTPUT = ROOT / 'runs/deepseek-spora-boundary-representability-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if runtime.sha(SOURCE) != SOURCE_SHA or runtime.sha(TRAIN) != TRAIN_SHA:
        raise ValueError('Source digest mismatch')
    prior = json.loads(INPUT.read_text(encoding='utf-8'))
    if prior['source_checkpoint_sha256'] != runtime.sha(runtime.CHECKPOINT):
        raise ValueError('Prior checkpoint mismatch')
    names = select_names(parquet.read_table(SOURCE).to_pylist(),
                         {r['text'] for r in parquet.read_table(TRAIN).to_pylist()})
    cases = []
    for digest, name in names:
        for group, subject in (('known', name), ('near_suffix', name + '子'), ('near_prefix', '小' + name)):
            cases.append((group, digest, subject, f'{subject}现在喜欢喝什么？'))
    tokenizer = AutoTokenizer.from_pretrained(runtime.MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    questions = [case[3] for case in cases]
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                           add_generation_prompt=True) for q in questions]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                        return_offsets_mapping=True)
    offsets = encoded['offset_mapping']
    masks = question_mask(offsets, encoded['attention_mask'], texts, questions)
    rows = []
    for i, (group, digest, subject, question) in enumerate(cases):
        previous = prior['rows'][i]
        qsha = hashlib.sha256(question.encode('utf-8')).hexdigest()
        if previous['group'] != group or previous['source_digest'] != digest or previous['question_sha256'] != qsha:
            raise ValueError('Previous example order or content mismatch')
        start = texts[i].find(question)
        gold = torch.zeros(len(question), dtype=torch.bool)
        gold[:len(subject)] = True
        token_mask = ((offsets[i, :, 0] < start + len(subject)) & (offsets[i, :, 1] > start) & masks[i])
        reconstructed = character_mask(offsets[i], token_mask, start, len(question))
        rows.append({'group': group, 'question_sha256': qsha,
                     'gold_representable': bool(torch.equal(reconstructed, gold)),
                     'default_exact_subject': previous['default_exact_subject'],
                     'candidate_exact_subject': previous['candidate_exact_subject']})
    summary = {}
    for group in ('known', 'near_suffix', 'near_prefix'):
        subset = [row for row in rows if row['group'] == group]
        summary[group] = {'total': len(subset), 'representable': sum(r['gold_representable'] for r in subset),
                          'candidate_miss_representable': sum(r['gold_representable'] and not r['candidate_exact_subject'] for r in subset),
                          'candidate_miss_unrepresentable': sum(not r['gold_representable'] and not r['candidate_exact_subject'] for r in subset)}
    result = {'status': 'token_boundary_diagnostic', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': prior['source_checkpoint_sha256'],
              'limitations': ['Only tests whether token masks can express gold spans in the same synthetic 60 questions.',
                              'Does not measure memory reads, real dialogue, or model-level character localization.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
