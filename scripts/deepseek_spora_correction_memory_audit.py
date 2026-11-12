"""Audit correction semantics before involving the language adapter.

This deliberately stops after the continuous page read. It does not generate
text, train weights, or alter the durable individual.
"""
import json
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_spora_semantic_bridge_pilot import CODEC_SLOT_COUNT
from spora.continuous_text_memory import ContinuousTextCodec


OUT = p.ROOT / 'runs/continuous-reader-correction-memory-audit-20261112'
SEED = 72381
WORDS = ('苹果', '香蕉', '西瓜', '葡萄', '草莓', '橘子', '桃子', '芒果',
         '梨子', '柠檬', '菠萝', '樱桃', '柚子', '荔枝', '榴莲', '蓝莓',
         '火龙果', '木瓜')
IDENTITY = '小赵'
KEY = 3


def encode(codec, tokenizer, embedding_device, value):
    tokens = [tokenizer.encode(IDENTITY, add_special_tokens=False)[-1]]
    tokens += tokenizer.encode(WORDS[value], add_special_tokens=False)
    return codec.encode(p.torch.tensor(tokens, dtype=p.torch.long, device=embedding_device)), tokens


def write_sequence(model, codec, tokenizer, device, values):
    fast, velocity = model.initial_state(1)
    for site in fast:
        site[0] = p.torch.zeros_like(site[0])
    targets = []
    for value in values:
        target, tokens = encode(codec, tokenizer, device, value)
        padded = p.torch.zeros((1, model.config.memory_width), device=device)
        padded[0, :target.numel()] = target
        with p.torch.no_grad():
            for index, site in enumerate(model.memories):
                fast[index], velocity[index] = site.write_temporal(
                    fast[index], velocity[index], site.current_basis[KEY:KEY + 1].to(device),
                    padded, p.torch.ones(1, device=device),
                    p.torch.zeros(1, dtype=p.torch.long, device=device), False)
        targets.append((value, padded, tokens))
    reads = []
    for index, site in enumerate(model.memories):
        read = site.retrieve([fast[index][0]], site.current_basis[KEY:KEY + 1].to(device))
        reads.append(read.detach().clone())
    return fast, targets, reads


def multi_entity_audit(model, codec, tokenizer, device):
    assignments = [(0, 0), (1, 1), (2, 3), (3, 9), (4, 16), (5, 17)]
    fast, velocity = model.initial_state(1)
    for site in fast:
        site[0] = p.torch.zeros_like(site[0])
    pages = []
    for key, value in assignments:
        identity = ('小林', '小周', '小王', '小赵', '小吴', '小郑')[key]
        tokens = [tokenizer.encode(identity, add_special_tokens=False)[-1]]
        tokens += tokenizer.encode(WORDS[value], add_special_tokens=False)
        code = codec.encode(p.torch.tensor(tokens, dtype=p.torch.long, device=device))
        target = p.torch.zeros((1, model.config.memory_width), device=device)
        target[0, :code.numel()] = code
        with p.torch.no_grad():
            for index, site in enumerate(model.memories):
                fast[index], velocity[index] = site.write_temporal(
                    fast[index], velocity[index], site.current_basis[key:key + 1].to(device),
                    target, p.torch.ones(1, device=device),
                    p.torch.zeros(1, dtype=p.torch.long, device=device), False)
        pages.append((key, value, tokens))
    results = []
    for key, value, tokens in pages:
        outputs = []
        for index, site in enumerate(model.memories):
            read = site.retrieve([fast[index][0]], site.current_basis[key:key + 1].to(device))
            decoded = codec.decode(read[0, :codec.slot_count * codec.slot_width])['token_ids']
            outputs.append({'decoded': decoded, 'expected': tokens + [tokenizer.eos_token_id],
                            'exact': decoded == tokens + [tokenizer.eos_token_id]})
        results.append({'key': key, 'value': WORDS[value], 'sites': outputs})
    return results


def vector_metrics(read, target, width):
    read = read[:, :width].float()
    target = target[:, :width].float()
    cosine = p.torch.nn.functional.cosine_similarity(read, target, dim=-1)
    return {'cosine_to_target': float(cosine[0]),
            'read_norm': float(read.norm()), 'target_norm': float(target.norm()),
            'mse': float((read - target).square().mean())}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    p.torch.manual_seed(SEED)
    p.torch.set_num_threads(2)
    p.torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    source_sha = p.sha(p.runtime.CHECKPOINT)
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    with tempfile.TemporaryDirectory() as temp:
        p.runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        model = chat.organism.eval()
        device = next(model.parameters()).device
        original_digest = p.frozen_digest(model, chat.state)
        embedding = chat.base.model.embed_tokens.weight.detach()
        codec = ContinuousTextCodec(embedding.cpu(), memory_width=model.config.memory_width,
                                    slot_count=CODEC_SLOT_COUNT, seed=29071,
                                    eos_token_id=tokenizer.eos_token_id).to(device)
        multi_codec = ContinuousTextCodec(embedding.cpu(), memory_width=model.config.memory_width,
                                          slot_count=5, seed=29071,
                                          eos_token_id=tokenizer.eos_token_id).to(device)
        width = codec.slot_count * codec.slot_width
        sequences = {'old_only': [0], 'new_only': [1], 'old_then_new': [0, 1],
                     'new_then_old': [1, 0], 'old_new_old_new': [0, 1, 0, 1]}
        rows = []
        for name, values in sequences.items():
            _, targets, reads = write_sequence(model, codec, tokenizer, device, values)
            final_value, final_target, final_tokens = targets[-1]
            decoded = []
            site_metrics = []
            for read in reads:
                decoded.append(codec.decode(read[0, :width])['token_ids'])
                metrics = vector_metrics(read, final_target, width)
                metrics['cosine_to_old'] = vector_metrics(read, targets[0][1], width)['cosine_to_target']
                metrics['cosine_to_new'] = vector_metrics(read, targets[-1][1], width)['cosine_to_target']
                site_metrics.append(metrics)
            expected_tokens = final_tokens + [tokenizer.eos_token_id]
            rows.append({'sequence': name, 'values': [WORDS[value] for value in values],
                         'expected_final_tokens': expected_tokens, 'decoded_tokens': decoded,
                         'decoded_matches_final': decoded[0] == expected_tokens and decoded[1] == expected_tokens,
                         'sites': site_metrics})
        multi_entity = multi_entity_audit(model, multi_codec, tokenizer, device)
        if p.frozen_digest(model, chat.state) != original_digest:
            raise ValueError('Original individual changed')
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    report = {'status': 'continuous_memory_correction_without_language_audit',
              'source_sha256': source_sha, 'codec': codec.config_dict(), 'key': KEY,
              'identity': IDENTITY, 'sequences': rows, 'multi_entity': multi_entity, 'width': width,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': p.torch.cuda.max_memory_allocated(),
              'limitations': ['Direct numeric address and controlled writes.',
                              'No language adapter or natural-language correction route.',
                              'Continuous code decoding is a diagnostic, not open memory semantics.',
                              'A high read cosine does not prove causal generation use.']}
    OUT.mkdir(parents=False, exist_ok=False)
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'sequences': len(rows), 'seconds': report['seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
