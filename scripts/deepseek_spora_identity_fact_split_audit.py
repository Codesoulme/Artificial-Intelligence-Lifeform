"""Audit identity and fact slot recovery independently from a saved bridge."""
import json
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_spora_semantic_bridge_pilot import TokenCodeDecoder
from spora.continuous_text_memory import ContinuousTextCodec


SOURCE = p.ROOT / 'runs/continuous-reader-text-reconstruction-pilot-20261113'
OUT = p.ROOT / 'runs/continuous-reader-identity-fact-split-audit-20261113'
NAMES = ('小林', '小周', '小王', '小赵')
WORDS = ('苹果', '香蕉', '西瓜', '葡萄', '草莓', '橘子', '桃子', '芒果',
         '梨子', '柠檬', '菠萝', '樱桃', '柚子', '荔枝', '榴莲', '蓝莓',
         '火龙果', '木瓜')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch = p.torch
    torch.manual_seed(92317)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    source_sha = p.sha(p.runtime.CHECKPOINT)
    payload = torch.load(SOURCE / 'reconstructor.pt', map_location='cpu', weights_only=True)
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    with tempfile.TemporaryDirectory() as temp:
        p.runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        model = chat.organism.eval()
        device = next(model.parameters()).device
        embedding = chat.base.model.embed_tokens.weight.detach()
        codec = ContinuousTextCodec(embedding.cpu(), memory_width=model.config.memory_width,
                                    slot_count=5, seed=29071,
                                    eos_token_id=tokenizer.eos_token_id).to(device)
        bridge = TokenCodeDecoder(codec.slot_width, embedding.shape[1]).to(device)
        bridge.load_state_dict(payload['bridge'], strict=True)
        bridge.eval()
        names = {key: tokenizer.encode(name, add_special_tokens=False)[-1]
                 for key, name in enumerate(NAMES)}
        word_ids = [tokenizer.encode(word, add_special_tokens=False) for word in WORDS]
        normalized_embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
        rows = []
        for key, name in enumerate(NAMES):
            for value, word in enumerate(WORDS):
                fast, velocity = model.initial_state(1)
                for site in fast:
                    site[0] = torch.zeros_like(site[0])
                ids = [names[key]] + word_ids[value]
                encoded = codec.encode(torch.tensor(ids, dtype=torch.long, device=device))
                target = torch.zeros((1, model.config.memory_width), device=device)
                target[0, :encoded.numel()] = encoded
                with torch.no_grad():
                    for index, site in enumerate(model.memories):
                        fast[index], velocity[index] = site.write_temporal(
                            fast[index], velocity[index], site.current_basis[key:key + 1].to(device),
                            target, torch.ones(1, device=device),
                            torch.zeros(1, dtype=torch.long, device=device), False)
                reads = [site.retrieve([fast[index][0]], site.current_basis[key:key + 1].to(device))
                         for index, site in enumerate(model.memories)]
                memory = ((reads[0] + reads[1]) * 0.5)[:, :4 * codec.slot_width]
                slots = torch.nn.functional.normalize(memory.reshape(1, 4, codec.slot_width), dim=-1)
                decoded = bridge(slots).float()[0]
                expected_tokens = [names[key], *word_ids[value]]
                slot_rows = []
                for position, token_id in enumerate(expected_tokens[:4]):
                    vector = torch.nn.functional.normalize(decoded[position], dim=-1)
                    target_vector = normalized_embedding[token_id]
                    scores = normalized_embedding @ vector
                    nearest = int(scores.argmax())
                    slot_rows.append({'position': position, 'token_id': token_id,
                                      'nearest_token_id': nearest,
                                      'nearest_text': tokenizer.decode([nearest]),
                                      'cosine': float((vector * target_vector).sum()),
                                      'rank': int((scores > scores[token_id]).sum()) + 1})
                rows.append({'name': name, 'word': word, 'identity_key': key,
                             'expected_tokens': expected_tokens, 'slots': slot_rows})
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    report = {'status': 'identity_fact_slot_split_audit', 'source_sha256': source_sha,
              'candidate_sha256': p.sha(SOURCE / 'reconstructor.pt'), 'rows': rows,
              'identity_rows': len(rows), 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Controlled numeric addresses and direct writes.',
                              'Bridge nearest-token probes diagnose representation; they are not open generation.',
                              'No text reconstructor or Qwen answer generation is used.',
                              'The finite vocabulary is inherited from the local tokenizer.']}
    OUT.mkdir(parents=False, exist_ok=False)
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'rows': len(rows), 'seconds': report['seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
