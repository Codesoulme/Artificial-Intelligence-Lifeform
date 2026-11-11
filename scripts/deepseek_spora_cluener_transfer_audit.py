"""Audit the existing name-span locator before and after SPORA replacement."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import pyarrow.parquet as parquet
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_cluener_span_audit import SOURCE, SOURCE_SHA, character_mask, select_rows
from scripts.deepseek_learned_name_span import OUTPUT as LOCATOR, single_span_mask
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_observation_transfer_audit import CHECKPOINT
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.structure_transfer import SPORALayerReplacement

OUTPUT = ROOT / 'runs/deepseek-spora-cluener-transfer-audit-20261111.json'


def evaluate(rows, scores, offsets, masks, texts):
    result = []
    for i, row in enumerate(rows):
        start = texts[i].find(row['text'])
        gold = torch.zeros(len(row['text']), dtype=torch.bool)
        for left, right in row['spans']:
            gold[left:right] = True
        selected = (scores[i] >= 0.5) & masks[i]
        predicted = character_mask(offsets[i], selected, start, len(row['text']))
        _, contiguous = single_span_mask(scores[i], masks[i])
        result.append({'group': row['group'], 'text': row['text'], 'spans': row['spans'],
                       'exact': bool(torch.equal(predicted, gold)),
                       'predicted_characters': int(predicted.sum()), 'gold_characters': int(gold.sum()),
                       'true_positive_characters': int((predicted & gold).sum()),
                       'single_contiguous': contiguous})
    summary = {}
    for group in ('single', 'multiple', 'none'):
        selected = [row for row in result if row['group'] == group]
        predicted = sum(row['predicted_characters'] for row in selected)
        gold = sum(row['gold_characters'] for row in selected)
        summary[group] = {'examples': len(selected), 'exact': sum(row['exact'] for row in selected),
                          'single_contiguous': sum(row['single_contiguous'] for row in selected),
                          'character_precision': sum(row['true_positive_characters'] for row in selected) / predicted if predicted else None,
                          'character_recall': sum(row['true_positive_characters'] for row in selected) / gold if gold else None}
    return {'summary': summary, 'rows': result}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    locator_file = LOCATOR / 'locator.pt'
    locator_report = LOCATOR / 'result.json'
    source = {str(path): sha(path) for path in (MODEL / 'model.safetensors', SOURCE, locator_file, locator_report, CHECKPOINT)}
    if source[str(MODEL / 'model.safetensors')] != WEIGHT_SHA or source[str(SOURCE)] != SOURCE_SHA:
        raise ValueError('Protected provenance mismatch')
    data = torch.load(locator_file, map_location='cpu', weights_only=True)
    report = json.loads(locator_report.read_text(encoding='utf-8'))
    if report['locator_sha256'] != source[str(locator_file)] or data['weight_sha256'] != WEIGHT_SHA:
        raise ValueError('Locator provenance mismatch')
    rows, population = select_rows(parquet.read_table(SOURCE).to_pylist())
    questions = [row['text'] for row in rows]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                            add_generation_prompt=True) for question in questions]
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False, return_offsets_mapping=True)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    vectors, masks = extract(base, tokenizer, questions, batch_size=2)
    locator = nn.Linear(vectors.shape[-1] * 2, 1).eval()
    locator.load_state_dict(data['state'], strict=True)
    token_vectors = torch.cat([F.layer_norm(vectors[:, i].float(), (vectors.shape[-1],)) for i in range(2)], dim=-1)
    with torch.no_grad():
        old_scores = locator(token_vectors).squeeze(-1).sigmoid()
    old_result = evaluate(rows, old_scores, encoded['offset_mapping'], masks, texts)
    decoder, organism, state = load_integrated(CHECKPOINT, base, MODEL, individual_device='cuda')
    del decoder, organism, state
    blocks = {i: layer for i, layer in enumerate(base.model.layers) if isinstance(layer, SPORALayerReplacement)}
    variant = replacement_feature_variant_sha256(MODEL, blocks)
    replaced, replaced_masks = extract(base, tokenizer, questions, batch_size=2)
    replaced_vectors = torch.cat([F.layer_norm(replaced[:, i].float(), (replaced.shape[-1],)) for i in range(2)], dim=-1)
    with torch.no_grad():
        new_scores = locator(replaced_vectors).squeeze(-1).sigmoid()
    new_result = evaluate(rows, new_scores, encoded['offset_mapping'], replaced_masks, texts)
    if any(sha(Path(path)) != digest for path, digest in source.items()):
        raise ValueError('Protected input changed')
    output = {'status': 'development_only_no_writes', 'dataset': 'xusenlin/clue-ner',
              'source_sha256': SOURCE_SHA, 'source_population': population,
              'locator_sha256': source[str(locator_file)], 'checkpoint_sha256': source[str(CHECKPOINT)],
              'replacement_layers': sorted(blocks), 'feature_variant_sha256': variant,
              'base': old_result, 'replaced': new_result, 'seconds': time.perf_counter() - started,
              'limitations': ['Forty selected news examples; CLUENER is not a memory-chat benchmark.',
                              'The locator was trained on original DeepSeek features and was not adapted here.',
                              'Public text may overlap base pretraining; no writing or memory reading was performed.',
                              'Entity routing, source trust, polarity and temporal meaning remain untested.']}
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name, value in (('base', old_result), ('replaced', new_result)):
        print(name, value['summary'], flush=True)


if __name__ == '__main__':
    main()
