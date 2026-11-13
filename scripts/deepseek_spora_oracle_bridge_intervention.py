"""Swap the learned bridge for exact-codebook embeddings during isolated inference."""
from dataclasses import replace
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_bridge_reload_audit as audit
from scripts.deepseek_spora_multirelation_broad_pilot import pilot as configuration
from spora.continuous_text_memory import ContinuousTextCodec

torch, p = audit.torch, audit.p
audit.SOURCE = ROOT / 'runs/continuous-reader-cross-attention-pilot-20261113'
audit.OUT = ROOT / 'runs/deepseek-spora-oracle-bridge-intervention-20261113'
audit.EXTRA_ONLY = True
audit.PARITY_VALUE_INDICES = (0, 1, 3, 9, 16, 17)
done = False


def oracle_embeddings(raw, codebook, embedding, min_norm=1e-5):
    """Map valid slots to normalized table vectors; keep empty slots zero."""
    if (not isinstance(raw, torch.Tensor) or raw.ndim != 3 or raw.shape[0] != 1
            or not isinstance(codebook, torch.Tensor) or codebook.ndim != 2
            or raw.shape[-1] != codebook.shape[-1]
            or not isinstance(embedding, torch.Tensor) or embedding.ndim != 2
            or embedding.shape[0] != codebook.shape[0] or min_norm < 0):
        raise ValueError('Invalid oracle bridge tensors')
    valid = raw.norm(dim=-1) > min_norm
    flattened = torch.nn.functional.normalize(raw.reshape(-1, raw.shape[-1]).float(), dim=-1)
    ids = (flattened @ codebook.float().T).argmax(dim=-1)
    exact = embedding.index_select(0, ids).float().reshape(1, raw.shape[1], -1)
    return torch.nn.functional.normalize(exact, dim=-1) * valid.unsqueeze(-1)


def observe(*, decoder, tokenizer, spec, cross_modules, group, name, expected, question):
    global done
    if done or expected != '苹果':
        return
    embedding = decoder.base.model.embed_tokens.weight.detach()
    codec = ContinuousTextCodec(embedding.cpu(), memory_width=90, slot_count=5,
                                seed=29071, eos_token_id=tokenizer.eos_token_id).to('cuda')
    bridge = decoder.base.model.layers[18].self_attn.q_proj.bridge
    original_forward = bridge.forward
    flags = [module.enabled for module in cross_modules]
    original_state = spec['memory_state']
    addresses = spec['memory_addresses']
    identity = tokenizer.encode(name, add_special_tokens=False)[-1]
    rows = []

    def oracle_forward(_bridge, codes):
        active = decoder._memory_context.get()
        if active is None:
            raise ValueError('Oracle bridge requires an explicit memory context')
        matrices, selected, mask, slots = active
        if selected is None or len(matrices) != 2 or codes.ndim != 3 or codes.shape[0] != 1:
            raise ValueError('Unsupported oracle bridge call')
        if mask is not None and not bool(mask[0]):
            raise ValueError('Unexpected masked bridge call')
        reads = []
        for matrix, address in zip(matrices, selected):
            key = torch.nn.functional.normalize(address.float(), dim=-1)
            reads.append(torch.bmm(matrix[0].float(), key.unsqueeze(-1)).squeeze(-1))
        raw = ((reads[0] + reads[1]) * 0.5).reshape(1, 5, 18)[:, :codes.shape[1]]
        return oracle_embeddings(raw, codec.codebook, embedding)

    def generate(text, memory):
        prompt = p.closed_think_inputs(tokenizer, text, 'cuda')
        with torch.inference_mode():
            output = decoder.generate(**prompt, **memory, use_cache=True, max_new_tokens=24,
                                      do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                      eos_token_id=tokenizer.eos_token_id)
        tail = output[0, prompt['input_ids'].shape[1]:].tolist()
        return {'answer': tokenizer.decode(tail, skip_special_tokens=True).strip(), 'token_ids': tail}

    try:
        for module in cross_modules:
            module.enabled = False
        for domain, (relation, questions, train, held) in configuration.DOMAINS.items():
            inquiry = questions[0].format(name=name)
            off = generate(inquiry, {})
            for partition, values in (('supervised', train[:2]), ('held', held)):
                for fact in values:
                    ids = tokenizer.encode(fact, add_special_tokens=False)
                    if not 1 <= len(ids) <= 3:
                        raise ValueError('Fact exceeds packet capacity')
                    with torch.no_grad():
                        packet = codec.encode(torch.tensor([identity, *ids], device='cuda')).reshape(1, -1)
                        matrices = []
                        for matrix, address in zip(original_state.current, addresses):
                            key = torch.nn.functional.normalize(address.float(), dim=-1)
                            old = torch.bmm(matrix.float(), key.unsqueeze(-1)).squeeze(-1)
                            scale = old.reshape(1, 5, 18)[:, 0].norm(dim=-1, keepdim=True)
                            matrices.append(matrix.float() + (packet * scale - old).unsqueeze(-1) * key.unsqueeze(1))
                        reads = [torch.bmm(matrix, torch.nn.functional.normalize(address.float(), dim=-1).unsqueeze(-1)).squeeze(-1)
                                 for matrix, address in zip(matrices, addresses)]
                        if codec.decode(torch.stack(reads).mean(0)[0])['token_ids'] != [identity, *ids, tokenizer.eos_token_id]:
                            raise ValueError('Packet did not recover its controlled tokens')
                    memory = {**spec, 'memory_state': replace(original_state, current=matrices)}
                    memory.pop('use_cache', None)
                    bridge.forward = original_forward
                    baseline = generate(inquiry, memory)
                    bridge.forward = types.MethodType(oracle_forward, bridge)
                    exact = generate(inquiry, memory)
                    bridge.forward = original_forward
                    def correct(result):
                        return result['answer'] in (fact, fact+'。', f'{name}{relation}{fact}。', f'{name}{relation}{fact}')
                    row = {'domain': domain, 'partition': partition, 'fact': fact, 'question': inquiry,
                           'codec_verified': True, 'off': {**off, 'strict_correct': correct(off)},
                           'learned': {**baseline, 'strict_correct': correct(baseline)},
                           'exact_embedding': {**exact, 'strict_correct': correct(exact)}}
                    rows.append(row)
                    with (audit.OUT/'intervention-rows.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                    print(json.dumps({'domain': domain, 'completed': len(rows)}), flush=True)
    finally:
        bridge.forward = original_forward
        for module, flag in zip(cross_modules, flags):
            module.enabled = flag
    summary = {partition: {mode: sum(row[mode]['strict_correct'] for row in rows if row['partition'] == partition)
                           for mode in ('off', 'learned', 'exact_embedding')}
               for partition in ('supervised', 'held')}
    result = {'summary': summary, 'rows': rows,
              'limitations': ['Source candidate and decoder adapters were not trained with the substituted bridge.',
                              'Exact embedding access depends on a fixed shared full-vocabulary codebook.',
                              'No token text is added to the question; answers still come from the decoder.',
                              'One person, one explicit address and one short synthetic fact at a time.',
                              'Groups refer to later broad-pilot values, not source-bridge training splits.',
                              'Strict answer scoring can miss valid paraphrases; inspect raw outputs.']}
    (audit.OUT/'intervention-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    done = True
    print(json.dumps(summary, ensure_ascii=False), flush=True)


audit.PROBE_OBSERVER = observe
if __name__ == '__main__':
    audit.main()
    if not done:
        raise ValueError('Oracle bridge intervention did not run')
