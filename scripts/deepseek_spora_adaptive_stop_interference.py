"""Compare forced irrelevant memory with the adaptive read head disabled."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_adaptive_stop_pilot import OUT as HEAD_DIR, SOURCE
from scripts.deepseek_spora_diverse_kl_correction_audit import load_reviewed_prompts
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from scripts.deepseek_spora_output_aware_writer import GAIN, norm_vectors
from spora.research_adaptive_semantic_head import AdaptiveSemanticHead, use_adaptive_slot
from spora.research_semantic_slot import SequenceSemanticSlotLMHead, use_slot_sequence
from spora.semantic_text_writer import TextSemanticWriter, value_positions

OUT = ROOT / 'runs/deepseek-spora-adaptive-stop-interference-20261116.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    head_report = json.loads((HEAD_DIR / 'result.json').read_text(encoding='utf-8'))
    head_file, writer_file = HEAD_DIR / 'adaptive-head.pt', SOURCE / 'writer.pt'
    head_sha, writer_sha = base.sha(head_file), base.sha(writer_file)
    if head_report['candidate_sha256'] != head_sha or head_report['writer_sha256'] != writer_sha:
        raise ValueError('Head or writer provenance mismatch')
    saved_head = torch.load(head_file, map_location='cpu', weights_only=True)
    saved_writer = torch.load(writer_file, map_location='cpu', weights_only=True)
    paths = {i: PAIR / f'layer-{i}.pt' for i in (6, 12)}
    hashes = {str(i): base.sha(path) for i, path in paths.items()}
    if (saved_head['base_sha256'] != base.WEIGHT_SHA or saved_head['writer_sha256'] != writer_sha
            or saved_head['layer_sha256'] != hashes):
        raise ValueError('Head binding mismatch')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed(skip_replaced=(6, 12)).eval()
    for i, path in paths.items():
        model.model.layers[i] = _load_candidate(path, i, model).eval()
    writer = TextSemanticWriter(model.config.hidden_size).to('cuda').eval()
    writer.load_state_dict(saved_writer['state'], strict=True)
    base_head = model.lm_head
    adaptive = AdaptiveSemanticHead(base_head, rank=saved_head['rank'], gain=saved_head['gain']).to('cuda').eval()
    incompatible = adaptive.load_state_dict(saved_head['state'], strict=False)
    if (set(incompatible.missing_keys) != {f'base.{key}' for key in base_head.state_dict()}
            or incompatible.unexpected_keys):
        raise ValueError('Adaptive head parameters are incomplete')
    one_shot = SequenceSemanticSlotLMHead(base_head, scale=GAIN)
    encoded, positions = value_positions(tokenizer, '小赵喜欢喝咖啡。', '咖啡')
    encoded = {k: v.to('cuda') for k, v in encoded.items()}
    with torch.inference_mode():
        hidden = model.model(**encoded, output_hidden_states=True, use_cache=False,
                             return_dict=True).hidden_states[1][0, positions].float()
        slot = norm_vectors(writer(hidden), saved_writer['train_norm'])[0]
    _, prompts = load_reviewed_prompts()
    rows = []
    for case in prompts:
        prompt = pilot.closed_think_inputs(tokenizer, case['prompt'], 'cuda')
        variants = {}
        with torch.inference_mode():
            for mode in ('base', 'adaptive_off', 'one_shot_forced', 'adaptive_forced'):
                model.lm_head = base_head if mode == 'base' else one_shot if mode == 'one_shot_forced' else adaptive
                if mode == 'one_shot_forced':
                    with use_slot_sequence(slot.unsqueeze(0)):
                        generated = model.generate(**prompt, use_cache=True, do_sample=False, max_new_tokens=24,
                                                   pad_token_id=tokenizer.pad_token_id,
                                                   eos_token_id=tokenizer.eos_token_id)
                elif mode == 'adaptive_forced':
                    with use_adaptive_slot(slot):
                        generated = model.generate(**prompt, use_cache=True, do_sample=False, max_new_tokens=24,
                                                   pad_token_id=tokenizer.pad_token_id,
                                                   eos_token_id=tokenizer.eos_token_id)
                else:
                    generated = model.generate(**prompt, use_cache=True, do_sample=False, max_new_tokens=24,
                                               pad_token_id=tokenizer.pad_token_id,
                                               eos_token_id=tokenizer.eos_token_id)
                tail = generated[0, prompt['input_ids'].shape[1]:].tolist()
                variants[mode] = {'ids': tail, 'text': tokenizer.decode(tail, skip_special_tokens=True).strip(),
                                  'ended': bool(tail and tail[-1] == tokenizer.eos_token_id)}
        rows.append({'source_group': case['group'], 'source_line': case['line'], 'variants': variants})
        print(json.dumps({'completed': len(rows), 'total': len(prompts)}), flush=True)
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA or base.sha(head_file) != head_sha
            or base.sha(writer_file) != writer_sha or any(base.sha(path) != hashes[str(i)] for i, path in paths.items())):
        raise ValueError('Protected input changed')
    summary = {'cases': len(rows),
               'off_same': sum(r['variants']['adaptive_off']['ids'] == r['variants']['base']['ids'] for r in rows),
               'one_shot_forced_changed': sum(r['variants']['one_shot_forced']['ids'] != r['variants']['base']['ids'] for r in rows),
               'adaptive_forced_changed': sum(r['variants']['adaptive_forced']['ids'] != r['variants']['base']['ids'] for r in rows),
               'adaptive_forced_ended': sum(r['variants']['adaptive_forced']['ended'] for r in rows)}
    report = {'summary': summary, 'rows': rows, 'base_sha256': base.WEIGHT_SHA, 'head_sha256': head_sha,
              'writer_sha256': writer_sha, 'layer_sha256': hashes, 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Eight previously reviewed ordinary questions, one unrelated forced coffee slot.',
                              'Forced intervention does not estimate real-world routing errors.',
                              'Token changes are not equivalent to quality changes; outputs are retained locally.',
                              '24-token cap can truncate; no autonomous relevance or open-chat acceptance.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
