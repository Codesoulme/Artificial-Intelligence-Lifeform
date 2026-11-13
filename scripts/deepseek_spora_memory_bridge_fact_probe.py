"""Measure the existing latent bridge against base embeddings for fact tokens."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_bridge_reload_audit as audit
from scripts.deepseek_spora_multirelation_broad_pilot import pilot as configuration
from spora.continuous_text_memory import ContinuousTextCodec

torch, p = audit.torch, audit.p
audit.SOURCE = ROOT / 'runs/continuous-reader-cross-attention-pilot-20261113'
audit.OUT = ROOT / 'runs/deepseek-spora-memory-bridge-fact-probe-20261113'
audit.EXTRA_ONLY = True
audit.PARITY_VALUE_INDICES = (0, 1, 3, 9, 16, 17)
done = False


def observe(*, decoder, tokenizer, spec, cross_modules, group, name, expected, question):
    global done
    if done or expected != '苹果':
        return
    embedding = decoder.base.model.embed_tokens.weight.detach()
    bridge = decoder.base.model.layers[18].self_attn.q_proj.bridge
    codec = ContinuousTextCodec(embedding.cpu(), memory_width=90, slot_count=5,
                                seed=29071, eos_token_id=tokenizer.eos_token_id).to('cuda')
    addresses = spec['memory_addresses']
    original = spec['memory_state']
    identity = tokenizer.encode(name, add_special_tokens=False)[-1]
    vectors, rows = [], []
    with torch.no_grad():
        for domain, (_, _, train, held) in configuration.DOMAINS.items():
            for partition, values in (('supervised_in_broad_pilot', train), ('held_in_broad_pilot', held)):
                for fact in values:
                    ids = tokenizer.encode(fact, add_special_tokens=False)
                    if not 1 <= len(ids) <= 3:
                        raise ValueError('Fact exceeds the packet capacity')
                    packet = codec.encode(torch.tensor([identity, *ids], device='cuda')).reshape(1, -1)
                    reads = []
                    for matrix, address in zip(original.current, addresses):
                        key = torch.nn.functional.normalize(address.float(), dim=-1)
                        old = torch.bmm(matrix.float(), key.unsqueeze(-1)).squeeze(-1)
                        scale = old.reshape(1, 5, 18)[:, 0].norm(dim=-1, keepdim=True)
                        updated = matrix.float() + (packet * scale - old).unsqueeze(-1) * key.unsqueeze(1)
                        reads.append(torch.bmm(updated, key.unsqueeze(-1)).squeeze(-1))
                    read = torch.stack(reads).mean(0)[0]
                    if codec.decode(read)['token_ids'] != [identity, *ids, tokenizer.eos_token_id]:
                        raise ValueError('Controlled packet failed to decode')
                    slots = read.reshape(5, 18)
                    for position, token_id in enumerate(ids):
                        code = torch.nn.functional.normalize(slots[position+1], dim=-1)
                        vector = bridge(code.reshape(1, 1, -1)).reshape(-1).float()
                        if not bool(torch.isfinite(vector).all()):
                            raise ValueError('Nonfinite bridge output')
                        vectors.append(vector)
                        rows.append({'domain': domain, 'partition': partition, 'fact': fact,
                                     'token_position': position, 'token_id': token_id})
        vectors = torch.stack(vectors)
        # All target scores and competitor scores use the same chunked matmul,
        # avoiding inconsistent rounding in the rank comparisons.
        chunks = []
        for start in range(0, embedding.shape[0], 4096):
            chunk = torch.nn.functional.normalize(embedding[start:start+4096].float(), dim=-1)
            chunks.append(vectors @ chunk.T)
        scores = torch.cat(chunks, dim=1)
        targets = torch.tensor([row['token_id'] for row in rows], device='cuda')
        wanted = scores.gather(1, targets[:, None]).squeeze(1)
        top = scores.argmax(dim=1)
        ranks = (scores > wanted[:, None]).sum(dim=1) + 1
        ties = (scores == wanted[:, None]).sum(dim=1)
        for i, row in enumerate(rows):
            row.update({'cosine': float(wanted[i]), 'rank': int(ranks[i]), 'ties': int(ties[i]),
                        'unique_top1': int(ranks[i]) == 1 and int(ties[i]) == 1,
                        'nearest_token_id': int(top[i]),
                        'nearest_token_text': tokenizer.decode([int(top[i])], skip_special_tokens=False)})
    summary = {}
    for partition in ('supervised_in_broad_pilot', 'held_in_broad_pilot'):
        chosen = [row for row in rows if row['partition'] == partition]
        summary[partition] = {'tokens': len(chosen),
                              'unique_top1': sum(row['unique_top1'] for row in chosen),
                              'top10': sum(row['rank'] <= 10 for row in chosen),
                              'mean_cosine': sum(row['cosine'] for row in chosen) / len(chosen)}
    result = {'summary': summary, 'rows': rows, 'vocab_size': int(embedding.shape[0]),
              'limitations': ['The exact shared codec lookup and explicitly addressed synthetic packets are controls.',
                              'Bridge was trained on random base tokens; overlap with these values is unknown.',
                              'Embedding reconstruction is not language generation or a causal bottleneck proof.',
                              'Memory states are isolated and do not represent a long-lived individual.']}
    (audit.OUT/'bridge-fact-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    done = True
    print(json.dumps(summary, ensure_ascii=False), flush=True)


audit.PROBE_OBSERVER = observe
if __name__ == '__main__':
    audit.main()
    if not done:
        raise ValueError('Bridge probe did not run')
