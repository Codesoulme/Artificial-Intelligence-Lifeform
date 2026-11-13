"""Measure actual fast-memory interference on isolated structured packets."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch.nn import functional as F
from scripts import deepseek_spora_long_memory_generate_parity as p
from spora.structured_text_memory import StructuredTextMemoryCodec

OUTPUT = ROOT / 'runs/deepseek-spora-structured-lane-interference-audit-20261113.json'
PREVIOUS = ROOT / 'runs/continuous-embedding-bridge-pilot-20261112/result.json'
STAGES = (1, 4, 8, 16, 32, 64)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    tokens = json.loads(PREVIOUS.read_text(encoding='utf-8'))
    if tokens['base_sha256'] != p.WEIGHT_SHA:
        raise ValueError('Token split does not match base')
    ids = tokens['train_ids'] + tokens['held_ids']
    if len(ids) != 320 or len(set(ids)) != 320:
        raise ValueError('Unexpected token split')
    torch.manual_seed(51904)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    source_sha = p.sha(p.runtime.CHECKPOINT)
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id in ids[:128]:
        raise ValueError('Unexpected EOS in data')
    with tempfile.TemporaryDirectory() as temp:
        p.runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        model = chat.organism.eval()
        if model.config.memory_width != 90 or model.config.key_count < max(STAGES):
            raise ValueError('Unexpected memory dimensions')
        device = next(model.parameters()).device
        original = p.frozen_digest(model, chat.state)
        codec = StructuredTextMemoryCodec(chat.base.model.embed_tokens.weight.detach().cpu(),
                                           memory_width=90, identity_slots=3, fact_slots=5,
                                           identity_slot_width=15, fact_slot_width=9, seed=29071,
                                           eos_token_id=tokenizer.eos_token_id).to(device)
        fast, velocity = model.initial_state(1)
        for site in fast:
            site[0] = torch.zeros_like(site[0])
        packets, metadata = [], []
        stages = []
        for key in range(max(STAGES)):
            packet, meta = codec.encode([ids[key]], [ids[key + 64]])
            packets.append(packet.detach().to(device))
            metadata.append(meta)
            with torch.no_grad():
                for index, site in enumerate(model.memories):
                    basis = site.current_basis[key:key + 1].to(device)
                    fast[index], velocity[index] = site.write_temporal(
                        fast[index], velocity[index], basis, packet.reshape(1, -1).to(device),
                        torch.ones(1, device=device), torch.zeros(1, dtype=torch.long, device=device), False)
            count = key + 1
            if count not in STAGES:
                continue
            records = []
            for address in range(count):
                with torch.no_grad():
                    vectors = [site.retrieve([fast[index][0]], site.current_basis[address:address + 1].to(device))[0]
                               for index, site in enumerate(model.memories)]
                    read = torch.stack(vectors).mean(0)
                    decoded = codec.decode(read, metadata[address])
                    actual = F.normalize(read[45:54].float(), dim=0)
                    expected = packets[address][45:54].float()
                    norm = float((actual - expected).norm())
                records.append({'key': address, 'identity_exact': decoded['identity']['token_ids'] == [ids[address]],
                                'fact_exact': decoded['fact']['token_ids'] == [ids[address + 64]],
                                'fact_first_code_error_l2': norm})
            errors = torch.tensor([row['fact_first_code_error_l2'] for row in records])
            stage = {'writes': count, 'identity_exact': sum(r['identity_exact'] for r in records),
                     'fact_exact': sum(r['fact_exact'] for r in records),
                     'fact_code_error_mean': float(errors.mean()), 'fact_code_error_max': float(errors.max()),
                     'fact_code_error_p90': float(torch.quantile(errors, .9)), 'records': records}
            stages.append(stage)
            print(json.dumps({k: v for k, v in stage.items() if k != 'records'}), flush=True)
        if p.frozen_digest(model, chat.state) != original:
            raise ValueError('Original state changed')
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    result = {'status': 'isolated_matrix_interference_probe', 'stages': stages,
              'codec': codec.config_dict(), 'base_sha256': p.WEIGHT_SHA, 'source_sha256': source_sha,
              'token_source_sha256': p.sha(PREVIOUS), 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Zero-initialized temporary matrices, sequential fixed numeric basis addresses.',
                              'One identity and one fact token plus EOS, selected from existing development vocabulary.',
                              'Shared full-vocabulary codebook and provided metadata; no free language generation.',
                              'No natural routing, real user state, repeated corrections, or real-world semantic validation.',
                              'Normalizing slot errors makes them comparable to unit-code perturbations, not identical noise distributions.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'seconds': result['seconds'], 'peak': result['peak_allocated_bytes']}), flush=True)


if __name__ == '__main__':
    main()
