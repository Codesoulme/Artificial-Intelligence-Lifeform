"""Train a small continuous-memory-to-embedding reconstruction bridge.

The bridge is isolated from the decoder. It tests whether the 90-dimensional
memory code can preserve token identity before language generation is involved.
No production reader, checkpoint, or individual state is changed.
"""
import json
import random
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p

OUT = p.ROOT / 'runs/continuous-embedding-bridge-pilot-20261112'
SEED = 66531
SLOTS = 4
STEPS = 400


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch = p.torch
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    source_sha = p.sha(p.runtime.CHECKPOINT)
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    with tempfile.TemporaryDirectory() as temp:
        p.runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        base, organism = chat.base.eval(), chat.organism.eval()
        embedding = base.model.embed_tokens.weight.detach().float()
        device = embedding.device
        if embedding.ndim != 2 or embedding.shape[1] < 32:
            raise ValueError('Unexpected embedding shape')
        memory_width = organism.config.memory_width
        slot_width = memory_width // SLOTS
        generator = torch.Generator(device='cpu').manual_seed(SEED)
        initial_projection = torch.randn((embedding.shape[1], slot_width), generator=generator,
                                         dtype=torch.float32, device='cpu').to(device)
        projection = torch.nn.Parameter(initial_projection.clone())
        decoder = torch.nn.Linear(memory_width, embedding.shape[1], bias=False, device=device)
        torch.nn.init.normal_(decoder.weight, std=0.02)
        # Use tokenizer-stable CJK tokens so the held test is human-readable and
        # not accidentally a multi-token decomposition mismatch.
        pool = []
        for token in range(tokenizer.vocab_size):
            text = tokenizer.decode([token], skip_special_tokens=False)
            if ('\u4e00' <= text[:1] <= '\u9fff' and
                    tokenizer.encode(text, add_special_tokens=False) == [token]):
                pool.append(token)
        if len(pool) < 300:
            raise ValueError('Insufficient stable CJK token pool')
        random.Random(SEED).shuffle(pool)
        train_ids = torch.tensor(pool[:256], dtype=torch.long, device=device)
        held_ids = torch.tensor(pool[256:320], dtype=torch.long, device=device)
        target_train = embedding.index_select(0, train_ids)
        target_held = embedding.index_select(0, held_ids)
        optimizer = torch.optim.AdamW([projection, *decoder.parameters()], lr=3e-4, weight_decay=1e-5,
                                      foreach=False)

        def codes(ids):
            projected = torch.nn.functional.normalize(embedding.index_select(0, ids) @ projection, dim=-1)
            result = torch.zeros((ids.numel(), memory_width), device=device)
            result[:, :slot_width] = projected
            return result

        def metrics(ids, targets):
            with torch.no_grad():
                predicted = decoder(codes(ids))
                cosine = torch.nn.functional.cosine_similarity(predicted, targets, dim=-1)
                # Rank only against the held/train candidate set; this avoids
                # claiming an open-vocabulary decoder when only a finite probe ran.
                candidates = torch.cat((target_train, target_held), dim=0)
                # Compare each target against the same finite candidate set.
                exact = []
                for row, target in zip(predicted, targets):
                    all_scores = torch.nn.functional.normalize(candidates, dim=-1) @ torch.nn.functional.normalize(row, dim=0)
                    exact.append(int((all_scores > (torch.nn.functional.normalize(target, dim=0) @ torch.nn.functional.normalize(row, dim=0))).sum()) + 1)
                return {'cosine_mean': float(cosine.mean()), 'cosine_min': float(cosine.min()),
                        'candidate_rank_mean': sum(exact) / len(exact), 'candidate_rank_top1': sum(x == 1 for x in exact),
                        'count': len(exact)}

        before = {'train': metrics(train_ids, target_train), 'held': metrics(held_ids, target_held)}
        trace = []
        for step in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            predicted = decoder(codes(train_ids))
            cosine_loss = 1 - torch.nn.functional.cosine_similarity(predicted, target_train, dim=-1).mean()
            scale = target_train.detach().norm(dim=-1).mean()
            mse_loss = ((predicted - target_train) / scale).square().mean()
            loss = cosine_loss + 0.25 * mse_loss
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite bridge loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([projection, *decoder.parameters()], 5.0, error_if_nonfinite=True)
            optimizer.step()
            if step == 0 or (step + 1) % 50 == 0:
                trace.append({'step': step + 1, 'loss': float(loss.detach()),
                              'cosine_loss': float(cosine_loss.detach()), 'grad_norm': float(norm),
                              'encoder_grad_norm': float(projection.grad.norm())})
                print(json.dumps(trace[-1]), flush=True)
        after = {'train': metrics(train_ids, target_train), 'held': metrics(held_ids, target_held)}
        candidate = {'projection': projection.detach().cpu(), 'decoder': decoder.state_dict(),
                     'source_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA}
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected source changed')
    OUT.mkdir(parents=False, exist_ok=False)
    torch.save(candidate, OUT / 'bridge.pt')
    report = {'before': before, 'after': after, 'trace': trace, 'train_ids': train_ids.tolist(),
              'held_ids': held_ids.tolist(), 'memory_width': memory_width, 'slot_width': slot_width,
              'steps': STEPS, 'source_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA,
              'candidate_sha256': p.sha(OUT / 'bridge.pt'), 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Embedding reconstruction is not language-model generation.',
                              'A finite stable CJK token probe, not open text or natural memory writes.',
                              'Only the first memory slot is used; no interference or routing test.',
                              'Candidate bridge is isolated and not installed in the reader.']}
    with (OUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'before': before, 'after': after, 'seconds': report['seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
