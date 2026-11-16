"""Reload the adaptive read head and evaluate unforced stopping."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_text_writer_broad_generalization as prior
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_adaptive_stop_pilot import OUT as HEAD_DIR, SOURCE
from scripts.deepseek_spora_broad_writer_phrases import VALUES
from scripts.deepseek_spora_diverse_kl_correction_audit import load_reviewed_prompts
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from scripts.deepseek_spora_output_aware_writer import GAIN, norm_vectors
from spora.research_adaptive_semantic_head import AdaptiveSemanticHead, use_adaptive_slot
from spora.research_semantic_slot import SequenceSemanticSlotLMHead, use_slot_sequence
from spora.semantic_text_writer import TextSemanticWriter, value_positions

OUT = ROOT / 'runs/deepseek-spora-adaptive-stop-reload-20261116.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    head_report = json.loads((HEAD_DIR / 'result.json').read_text(encoding='utf-8'))
    source_report = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    head_file, writer_file = HEAD_DIR / 'adaptive-head.pt', SOURCE / 'writer.pt'
    head_sha, writer_sha = base.sha(head_file), base.sha(writer_file)
    if (head_report['candidate_sha256'] != head_sha or head_report['writer_sha256'] != writer_sha
            or source_report['writer_sha256'] != writer_sha or head_report['base_sha256'] != base.WEIGHT_SHA):
        raise ValueError('Head or writer provenance mismatch')
    saved_head = torch.load(head_file, map_location='cpu', weights_only=True)
    saved_writer = torch.load(writer_file, map_location='cpu', weights_only=True)
    paths = {i: PAIR / f'layer-{i}.pt' for i in (6, 12)}
    hashes = {str(i): base.sha(path) for i, path in paths.items()}
    if (saved_head['base_sha256'] != base.WEIGHT_SHA or saved_head['writer_sha256'] != writer_sha
            or saved_head['layer_sha256'] != hashes or head_report['layer_sha256'] != hashes):
        raise ValueError('Head binding mismatch')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed(skip_replaced=(6, 12)).eval()
    for i, path in paths.items():
        model.model.layers[i] = _load_candidate(path, i, model).eval()
    if any(p.is_meta for p in model.parameters()):
        raise ValueError('Unmaterialized model parameter')
    writer = TextSemanticWriter(model.config.hidden_size).to('cuda').eval()
    writer.load_state_dict(saved_writer['state'], strict=True)
    base_head = model.lm_head
    adaptive = AdaptiveSemanticHead(base_head, rank=saved_head['rank'], gain=saved_head['gain']).to('cuda').eval()
    incompatible = adaptive.load_state_dict(saved_head['state'], strict=False)
    if (set(incompatible.missing_keys) != {f'base.{key}' for key in base_head.state_dict()}
            or incompatible.unexpected_keys):
        raise ValueError('Adaptive head parameters are incomplete')
    one_shot = SequenceSemanticSlotLMHead(base_head, scale=GAIN)
    train_ids = {tokenizer.encode(v, add_special_tokens=False)[0] for v in source_report['train_values'][:256]}
    rows = []
    for name in prior.HELD_NAMES:
        prompt = pilot.closed_think_inputs(tokenizer, f'{name}喜欢喝什么饮料？', 'cuda')
        for group, values in (('held_single', source_report['held_values']), ('phrases', VALUES)):
            for value in values:
                encoded, positions = value_positions(tokenizer, f'{name}喜欢喝{value}。', value)
                encoded = {k: v.to('cuda') for k, v in encoded.items()}
                target = encoded['input_ids'][0, positions].tolist()
                with torch.inference_mode():
                    hidden = model.model(**encoded, output_hidden_states=True, use_cache=False,
                                         return_dict=True).hidden_states[1][0, positions].float()
                    token_slots = norm_vectors(writer(hidden), saved_writer['train_norm'])
                    # One vector is retained throughout; no target-length or EOS slot is supplied.
                    slot = token_slots.mean(dim=0)
                    outputs = {}
                    for mode, head in (('one_shot', one_shot), ('adaptive', adaptive)):
                        model.lm_head = head
                        manager = use_slot_sequence(slot.unsqueeze(0)) if mode == 'one_shot' else use_adaptive_slot(slot)
                        with manager:
                            generated = model.generate(**prompt, use_cache=True, do_sample=False, max_new_tokens=16,
                                                       pad_token_id=tokenizer.pad_token_id,
                                                       eos_token_id=tokenizer.eos_token_id)
                        tail = generated[0, prompt['input_ids'].shape[1]:].tolist()
                        outputs[mode] = {'ids': tail, 'text': tokenizer.decode(tail, skip_special_tokens=True).strip(),
                                         'first_correct': bool(tail and tail[0] == target[0]),
                                         'ended': bool(tail and tail[-1] == tokenizer.eos_token_id),
                                         'exact': tail == target + [tokenizer.eos_token_id]}
                rows.append({'group': group, 'name': name, 'value': value, 'target': target,
                             'train_target_overlap': [token for token in target if token in train_ids],
                             'outputs': outputs})
            print(json.dumps({'name': name, 'group': group, 'cases': len(rows)}, ensure_ascii=True), flush=True)
    _, held_prompts = load_reviewed_prompts()
    ordinary = []
    for case in held_prompts:
        prompt = pilot.closed_think_inputs(tokenizer, case['prompt'], 'cuda')
        result = {}
        with torch.inference_mode():
            for mode, head in (('base', base_head), ('adaptive_off', adaptive)):
                model.lm_head = head
                generated = model.generate(**prompt, use_cache=True, do_sample=False, max_new_tokens=24,
                                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                result[mode] = generated[0, prompt['input_ids'].shape[1]:].tolist()
        ordinary.append({'source_group': case['group'], 'source_line': case['line'],
                         'same_tokens': result['base'] == result['adaptive_off']})
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA or base.sha(head_file) != head_sha
            or base.sha(writer_file) != writer_sha or any(base.sha(path) != hashes[str(i)] for i, path in paths.items())):
        raise ValueError('Protected input changed')
    summary = {}
    for group in ('held_single', 'phrases'):
        subset = [r for r in rows if r['group'] == group]
        summary[group] = {'cases': len(subset), **{
            mode: {key: sum(r['outputs'][mode][key] for r in subset) for key in ('first_correct', 'ended', 'exact')}
            for mode in ('one_shot', 'adaptive')}}
    result = {'summary': summary, 'rows': rows, 'ordinary': ordinary,
              'base_sha256': base.WEIGHT_SHA, 'writer_sha256': writer_sha, 'head_sha256': head_sha,
              'layer_sha256': hashes, 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two names, 64 synthetic held single-token values and eight previously used phrases.',
                              'One averaged slot for multi-token values; factual spans and identities are supplied.',
                              'No EOS vector or output step count is supplied; 16-token cap may truncate.',
                              'Held examples are developer-inspected; not independent production evaluation.',
                              'No persistent memory, autonomous writes, true multi-turn or open chat.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
