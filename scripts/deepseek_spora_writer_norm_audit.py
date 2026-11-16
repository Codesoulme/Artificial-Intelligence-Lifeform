"""Diagnose learned writer magnitude and full-vocabulary generation effects."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_text_writer_broad_generalization as prior
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_writer_layer_generation import SOURCE
from spora.research_semantic_slot import install_sequence_scaled, use_slot_sequence
from spora.semantic_text_writer import TextSemanticWriter, value_positions

OUT = ROOT / 'runs/deepseek-spora-writer-norm-audit-20261116.json'


def normalize_to_training_norm(vector, training_norm):
    if (not isinstance(vector, torch.Tensor) or vector.ndim != 1 or not bool(torch.isfinite(vector).all())
            or not isinstance(training_norm, (float, int)) or not 0 < training_norm < float('inf')):
        raise ValueError('Expected finite vector and positive training norm')
    norm = float(vector.norm())
    if not 0 < norm < float('inf'):
        raise ValueError('Cannot normalize zero or nonfinite vector')
    return vector * (training_norm / norm)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    source_report = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    source = SOURCE / 'writers.pt'
    source_sha = base.sha(source)
    if source_report['base_sha256'] != base.WEIGHT_SHA or source_report['weights_sha256'] != source_sha:
        raise ValueError('Writer provenance mismatch')
    payload = torch.load(source, map_location='cpu', weights_only=True)
    if payload['base_sha256'] != base.WEIGHT_SHA or set(payload['train_values']) & set(payload['held_values']):
        raise ValueError('Invalid writer token splits')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    writer = TextSemanticWriter(model.config.hidden_size).to('cuda').eval()
    writer.load_state_dict(payload['weights']['93471:shallow_layer_1'], strict=True)
    train_ids = [tokenizer.encode(value, add_special_tokens=False)[0] for value in payload['train_values']]
    with torch.no_grad():
        train_norm = float(model.model.embed_tokens.weight[train_ids].float().norm(dim=-1).mean())
    install_sequence_scaled(model, scale=32.)
    rows = []
    for person in prior.HELD_NAMES:
        prompt = pilot.closed_think_inputs(tokenizer, f'{person}喜欢喝什么饮料？', 'cuda')
        with torch.inference_mode():
            prompt_hidden = model.model(**prompt, use_cache=False, return_dict=True).last_hidden_state[:, -1:]
        for value in payload['held_values'][:8]:
            encoded, positions = value_positions(tokenizer, f'{person}喜欢喝{value}。', value)
            if len(positions) != 1:
                raise ValueError('Expected one value token')
            encoded = {k: v.to('cuda') for k, v in encoded.items()}
            token = int(encoded['input_ids'][0, positions[0]])
            with torch.inference_mode():
                fact = model.model(**encoded, use_cache=False, output_hidden_states=True, return_dict=True)
                vector = writer(fact.hidden_states[1][0, positions[0]].float())
                target = model.model.embed_tokens.weight[token].float()
                eos = model.model.embed_tokens.weight[tokenizer.eos_token_id].float()
                variants = {'raw': vector, 'training_norm': normalize_to_training_norm(vector, train_norm),
                            'direct': target}
                scores = {}
                for mode, slot in variants.items():
                    logits = model.lm_head.base(prompt_hidden + (32 * slot).to(prompt_hidden.dtype)).float()[0, -1]
                    rank = int((logits > logits[token]).sum()) + 1
                    with use_slot_sequence(torch.stack((slot, eos))):
                        output = model.generate(**prompt, use_cache=True, do_sample=False, max_new_tokens=4,
                                                pad_token_id=tokenizer.pad_token_id,
                                                eos_token_id=tokenizer.eos_token_id)
                    tail = output[0, prompt['input_ids'].shape[1]:].tolist()
                    scores[mode] = {'rank': rank, 'top1': int(logits.argmax()), 'generated_ids': tail,
                                    'exact': tail == [token, tokenizer.eos_token_id]}
                rows.append({'person': person, 'value': value, 'target_token': token,
                             'writer_norm': float(vector.norm()), 'target_norm': float(target.norm()),
                             'cosine': float(F.cosine_similarity(vector, target, dim=0)), 'scores': scores})
        print(json.dumps({'completed_person': person, 'cases': len(rows)}, ensure_ascii=False), flush=True)
    if base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA or base.sha(source) != source_sha:
        raise ValueError('Protected source changed')
    summary = {mode: {'first_rank_1': sum(r['scores'][mode]['rank'] == 1 for r in rows),
                      'full_exact': sum(r['scores'][mode]['exact'] for r in rows)}
               for mode in ('raw', 'training_norm', 'direct')}
    report = {'base_sha256': base.WEIGHT_SHA, 'writer_sha256': source_sha, 'training_norm': train_norm,
              'rows': rows, 'summary': summary, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two held names, eight synthetic single-character held values.',
                              'Value boundaries and read routing are manually supplied.',
                              'Full vocabulary score does not imply semantic or multi-token correctness.',
                              'Direct target and EOS vector are oracle controls, not learned memory.',
                              'Frozen unmodified base; no SPORA replacement or real individual.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'summary': summary, 'training_norm': train_norm}), flush=True)


if __name__ == '__main__':
    main()
