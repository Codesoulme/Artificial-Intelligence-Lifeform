"""Measure real delta-write interference with controlled addresses and text codes."""
import json
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_spora_continuous_text_capacity import sampled_sequences
from spora.continuous_text_memory import ContinuousTextCodec

OUT = p.ROOT / 'runs/continuous-memory-address-audit-20261112.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    t = p.torch
    t.set_num_threads(2)
    t.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    checkpoint = p.sha(p.runtime.CHECKPOINT)
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    examples, pool = sampled_sequences(tokenizer)
    with tempfile.TemporaryDirectory() as tmp:
        p.runtime.WORKING_CHECKPOINT = Path(tmp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        model = chat.organism.eval()
        original = p.frozen_digest(model, chat.state)
        device = next(model.parameters()).device
        if model.config.key_count < 32:
            raise ValueError('Need 32 separate addresses')
        codec = ContinuousTextCodec(chat.base.model.embed_tokens.weight.detach(),
                                    memory_width=model.config.memory_width, slot_count=4,
                                    seed=29071, eos_token_id=tokenizer.eos_token_id)
        width = codec.slot_count * codec.slot_width
        gate = t.ones(1, device=device)
        mode = t.zeros(1, dtype=t.long, device=device)
        targets = []
        for item in examples[:17]:
            target = t.zeros((1, model.config.memory_width), device=device)
            target[0, :width] = codec.encode(t.tensor(item['token_ids'], dtype=t.long, device=device))
            targets.append(target)
        rows = []
        for init in ('zero', 'model_initial'):
            fast, velocity = model.initial_state(1)
            if init == 'zero':
                for site in fast:
                    site[0] = t.zeros_like(site[0])
            initial = [site[0].detach().clone() for site in fast]
            def write(key, target):
                for i, site in enumerate(model.memories):
                    fast[i], velocity[i] = site.write_temporal(fast[i], velocity[i],
                        site.current_basis[key:key+1].to(device), target, gate, mode, False)
            def record(site_i, kind, key, address, expected=None):
                site = model.memories[site_i]
                with t.no_grad():
                    vector = site.retrieve([fast[site_i][0]], address)[0]
                    initial_vector = site.retrieve([initial[site_i]], address)[0]
                    norm = float(vector.norm())
                    decoded = codec.decode(vector[:width])
                    slot = vector[:codec.slot_width]
                    scores = codec.codebook @ t.nn.functional.normalize(slot, dim=0)
                    top = scores.topk(2)
                    result = {'init': init, 'site': site_i, 'kind': kind, 'key': key,
                              'norm': norm, 'initial_norm': float(initial_vector.norm()),
                              'decoded': decoded['token_ids'], 'top_score': float(top.values[0]),
                              'margin': float(top.values[0] - top.values[1])}
                    if expected is not None:
                        result.update(expected=examples[expected]['token_ids'] + [tokenizer.eos_token_id],
                                      exact=decoded['token_ids'] == examples[expected]['token_ids'] + [tokenizer.eos_token_id],
                                      target_l2_error=float((vector-targets[expected][0]).norm()))
                    rows.append(result)
            for key in range(16):
                write(key, targets[key])
            for i, site in enumerate(model.memories):
                for key in range(16):
                    address = site.current_basis[key:key+1].to(device)
                    record(i, 'written', key, address, key)
                for key in range(16, 32):
                    address = site.current_basis[key:key+1].to(device)
                    record(i, 'unwritten', key, address)
                a = site.current_basis[0:1].to(device)
                b = site.current_basis[1:2].to(device)
                for mix in (0.05, 0.2, 0.5):
                    address = t.nn.functional.normalize((1-mix)*a + mix*b, dim=-1)
                    record(i, f'mixed_{mix}', 0, address, 0)
            write(0, targets[16])
            for i, site in enumerate(model.memories):
                for key in range(16):
                    record(i, 'after_correction', key, site.current_basis[key:key+1].to(device),
                           16 if key == 0 else key)
        if p.frozen_digest(model, chat.state) != original:
            raise ValueError('Original state changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected inputs changed')
    result = {'rows': rows, 'pool': pool, 'base_sha256': p.WEIGHT_SHA, 'checkpoint_sha256': checkpoint,
              'seconds': time.perf_counter()-started, 'peak_allocated_bytes': t.cuda.max_memory_allocated(),
              'limitations': ['Controlled numeric basis addresses bypass real name routing and authority checks.',
                              'Zero and model-initial matrices are separate fresh states, not the existing individual.',
                              'Nearest-neighbor codec is not model-generated language or proof of known/unknown decisions.',
                              'One seed, sixteen sequences, one correction; mixed vectors are artificial.',
                              'No historical page, persisted state or long-term real-world interference tested.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'seconds': result['seconds'], 'rows': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
