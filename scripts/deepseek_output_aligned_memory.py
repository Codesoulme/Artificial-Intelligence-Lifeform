"""Compare output-aligned continuous event representations under a frozen decoder."""

import argparse
import gc
import json
import time
from pathlib import Path

from deepseek_focus_value_position_20261120 import MODEL, ROOT, SOURCE, load_integrated, prefix, read, sha, torch, write
from scripts.spora_event_focus_audit_20261118 import fresh

OUT = ROOT / 'runs/deepseek-output-aligned-memory'
RANKS = (9, 45, 90)
SLOTS = 8
GAIN = 24.0
SEED = 82417
FACTS = (
    (2, 'drink', '金桔茉莉冷泡茶', '烘焙乌龙燕麦奶'),
    (5, 'drink', '桂花荔枝汽水', '紫苏青梅苏打'),
    (9, 'work', '纸质文献修复师', '古籍装帧技师'),
    (12, 'work', '水下地形测绘员', '钟表机芯检验员'),
    (16, 'place', '榆湾旧船坞', '白榆山观测站'),
    (21, 'place', '澄浦织造馆', '青屿河口码头'),
)
QUESTIONS = {
    'drink': ('记录里的当事人偏爱哪种饮品？请完整回答。', '根据记录，这位当事人喜欢喝'),
    'work': ('这份经历记录中的人从事什么工作？请完整回答。', '根据记录，这个人的职业是'),
    'place': ('那次经历里，当事人去了哪里？请完整回答。', '根据记录，这位当事人去了'),
}


def fingerprint(paths):
    return {str(path): sha(path) for path in paths}


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def read_bank(site, bank, address):
    if bank.shape[0] != SLOTS or bank.ndim != 3:
        raise ValueError('Unexpected bank dimensions')
    return site.retrieve([bank], address.expand(SLOTS, -1))


def write_bank(site, bank, address, vectors):
    keys = address.expand(SLOTS, -1)
    # Each slot uses the same entity key and a distinct native matrix.
    result, _ = site.write([bank], [torch.zeros_like(bank)], keys, vectors,
                           torch.ones(SLOTS, device=bank.device), False)
    return result[0].detach()


@torch.inference_mode()
def emit(decoder, tokenizer, start, residual=None, gain=GAIN):
    context = torch.tensor([start], dtype=torch.long, device='cuda')
    emitted = []
    for step in range(SLOTS + 2):
        scores = decoder(input_ids=context, use_cache=False).logits[0, -1].float()
        if residual is not None and step < len(residual):
            scores += gain * decoder.base.lm_head(residual[step].to(decoder.base.lm_head.weight.dtype)).float()
        token = int(scores.argmax())
        emitted.append(token)
        if token == tokenizer.eos_token_id:
            break
        context = torch.cat((context, context.new_tensor([[token]])), dim=1)
    return emitted, tokenizer.decode(emitted, skip_special_tokens=True).strip()


def summarize(rows):
    summary = {}
    for rank in ('off', *RANKS, 'full_reference'):
        chosen = [r for r in rows if r['rank'] == rank]
        pairs = [[r for r in chosen if r['key'] == key] for key, *_ in FACTS]
        summary[str(rank)] = {
            'exact_values': sum(r['exact_value'] for r in chosen), 'total': len(chosen),
            'exact_tokens_with_end': sum(r['exact_tokens'] for r in chosen),
            'original_and_corrected': sum(len(pair) == 2 and all(r['exact_value'] for r in pair) for pair in pairs),
            'pairs': len(pairs),
        }
    return summary


@torch.inference_mode()
def main(out, replay=False, compensate=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (out / ('reload.json' if replay else 'result.json')).exists() or (not replay and out.exists()):
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0] < 3.7 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory for the frozen model')
    torch.set_num_threads(2)
    started = time.perf_counter()
    paths = (SOURCE, MODEL / 'model.safetensors', MODEL / 'tokenizer.json',
             MODEL / 'tokenizer_config.json', Path(__file__))
    hashes = fingerprint(paths)
    if replay:
        plan = read(out / 'plan.json')
        reference = read(out / 'result.json')
        if (hashes != plan['sources'] or sha(out / 'banks.pt') != reference['banks_sha256']
                or plan['compensate_gain'] != compensate):
            raise ValueError('Replay source or matrix checkpoint changed')
    else:
        out.mkdir()
        write(out / 'plan.json', {'sources': hashes, 'facts': FACTS, 'questions': QUESTIONS,
                                  'gain': GAIN, 'ranks': RANKS, 'slots': SLOTS, 'seed': SEED,
                                  'compensate_gain': compensate})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, dtype=torch.float16,
        attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, old_state = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    del parent, old_state
    decoder.eval().requires_grad_(False)
    organism, initial = fresh()
    del initial
    site = organism.memories[0]
    width = organism.config.memory_width
    if width < max(RANKS):
        raise ValueError('Projection does not fit in the native matrix width')
    del organism
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    weight = base.lm_head.weight
    head_width = weight.shape[1]
    generator = torch.Generator().manual_seed(SEED)
    basis = torch.linalg.qr(torch.randn(head_width, max(RANKS), generator=generator)).Q.to('cuda')
    token_sequences = {}
    for key, relation, before, after in FACTS:
        for version, value in enumerate((before, after)):
            tokens = encode(tokenizer, value)
            if not 2 <= len(tokens) < SLOTS or set(tokens) & set(tokenizer.all_special_ids):
                raise ValueError('Expected a multi-token value that fits in the matrix slots: ' + value)
            token_sequences[key, version] = [*tokens, tokenizer.eos_token_id]
    if replay:
        banks = torch.load(out / 'banks.pt', map_location='cpu', weights_only=True)
    else:
        banks = {}
        for rank in RANKS:
            bank = torch.zeros(SLOTS, width, width)
            for version in range(2):
                for key, *_ in FACTS:
                    ids = token_sequences[key, version]
                    vectors = torch.nn.functional.normalize(weight[ids].float() @ basis[:, :rank], dim=-1)
                    target = torch.zeros(SLOTS, width)
                    target[:len(ids), :rank] = vectors.cpu()
                    bank = write_bank(site, bank, site.current_basis[key:key + 1], target)
                banks[f'{rank}:{version}'] = bank.clone()
        torch.save(banks, out / 'banks.pt')
    if set(banks) != {f'{rank}:{version}' for rank in RANKS for version in range(2)}:
        raise ValueError('Incomplete bank checkpoint')
    print('Loaded frozen decoder and native banks', flush=True)

    rows = []
    prompts = {relation: encode(tokenizer, prefix(tokenizer, question) + opening)
               for relation, (question, opening) in QUESTIONS.items()}
    disabled = {relation: emit(decoder, tokenizer, prompt) for relation, prompt in prompts.items()}
    with (out / ('reload-progress.jsonl' if replay else 'progress.jsonl')).open('x', encoding='utf8') as progress:
        for key, relation, before, after in FACTS:
            start = prompts[relation]
            address = site.current_basis[key:key + 1]
            for version, value in enumerate((before, after)):
                expected = token_sequences[key, version]
                for rank in ('off', *RANKS, 'full_reference'):
                    intrinsic = None
                    gain = GAIN
                    if rank == 'off':
                        tokens, text = disabled[relation]
                    elif rank == 'full_reference':
                        # An explicitly labeled ceiling using the uncompressed output directions.
                        residual = torch.nn.functional.normalize(weight[expected].float(), dim=-1)
                        tokens, text = emit(decoder, tokenizer, start, residual)
                    else:
                        stored = read_bank(site, banks[f'{rank}:{version}'], address).to('cuda')[:, :rank]
                        residual = torch.nn.functional.normalize(stored, dim=-1, eps=1e-4) @ basis[:, :rank].T
                        # Unoccupied positions must not amplify orthogonal write noise.
                        residual[stored.norm(dim=-1) < 0.2] = 0
                        scores = base.lm_head(residual[:len(expected)].to(weight)).float()
                        targets = torch.tensor(expected, device=scores.device)
                        target_scores = scores.gather(1, targets[:, None])
                        intrinsic = (1 + (scores > target_scores).sum(dim=1)).tolist()
                        del scores
                        if compensate:
                            gain *= (head_width / rank) ** 0.5
                        tokens, text = emit(decoder, tokenizer, start, residual, gain)
                    row = {'key': key, 'relation': relation, 'version': version, 'rank': rank,
                           'value': value, 'value_token_count': len(expected) - 1,
                           'token_ids': tokens, 'expected_tokens': expected, 'continuation': text,
                           'exact_value': text.rstrip('。.!！ ') == value,
                           'exact_tokens': tokens == expected, 'evidence_only_target_ranks': intrinsic,
                           'gain': gain}
                    rows.append(row)
                    progress.write(json.dumps(row, ensure_ascii=False) + '\n')
            progress.flush()
            print('Scored', key, relation, len(rows), flush=True)
    if fingerprint(paths) != hashes:
        raise ValueError('A protected source changed')
    summary = summarize(rows)
    if replay and (rows != reference['rows'] or summary != reference['summary']):
        raise ValueError('Independent generation replay differs')
    write(out / ('reload.json' if replay else 'result.json'), {
        'rows': rows, 'summary': summary, 'banks_sha256': sha(out / 'banks.pt'),
        'matrix_payload_bytes': sum(bank.numel() * bank.element_size() for bank in banks.values()),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'elapsed_seconds': time.perf_counter() - started, 'exact_replay': replay,
    })
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--compensate', action='store_true')
    args = parser.parse_args()
    main(args.out, args.replay, args.compensate)
