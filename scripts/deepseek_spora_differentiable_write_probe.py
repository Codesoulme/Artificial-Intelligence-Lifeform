"""Trace one answer loss into an isolated writer through the real DeepSeek hybrid."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_bridge_reload_audit as audit
from spora.research_differentiable_memory import DifferentiablePacketWriter, differentiable_memory_context

torch, p = audit.torch, audit.p
audit.SOURCE = ROOT / 'runs/continuous-reader-cross-attention-pilot-20261113'
audit.OUT = ROOT / 'runs/deepseek-spora-differentiable-write-gradient-v3-20261113'
audit.EXTRA_ONLY = True
audit.PARITY_VALUE_INDICES = (0, 1, 3, 9, 16, 17)
done = False


def observe(*, decoder, tokenizer, spec, cross_modules, group, name, expected, question):
    global done
    if done or expected != '苹果':
        return
    addresses = spec['memory_addresses']
    state = spec['memory_state']
    prompt = p.closed_think_inputs(tokenizer, question, 'cuda')
    suffix = tokenizer.encode(f'{name}喜欢的水果是', add_special_tokens=False)
    target_id = tokenizer.encode(expected, add_special_tokens=False)[0]
    prefix = torch.cat((prompt['input_ids'], torch.tensor([suffix], device='cuda')), dim=1)
    observation = decoder.base.model.embed_tokens.weight[target_id].detach().float().unsqueeze(0)
    writer = DifferentiablePacketWriter(observation.shape[1], state.current[0].shape[-1]).to('cuda')
    with torch.no_grad():
        reads = [torch.bmm(matrix.float(), address.float().unsqueeze(-1)).squeeze(-1)
                 for matrix, address in zip(state.current, addresses)]
        packet = sum(reads) / len(reads)
        writer.packet.weight.copy_(packet.T @ observation / observation.square().sum())
        slot_mask = packet.reshape(1, 5, 18).norm(dim=-1) > 1e-5
        active_mask = slot_mask.unsqueeze(-1).expand(-1, -1, 18).reshape_as(packet)
    matrices = [writer(torch.zeros_like(matrix), address, observation, active_mask=active_mask)
                for matrix, address in zip(state.current, addresses)]
    original_flags = [module.enabled for module in cross_modules]
    for module in cross_modules:
        module.enabled = False
    try:
        with differentiable_memory_context(decoder, matrices, addresses):
            logits = decoder.base(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                                  use_cache=False, logits_to_keep=1).logits[0, -1].float()
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0),
                                                     torch.tensor([target_id], device='cuda'))
            gradient, = torch.autograd.grad(loss, writer.packet.weight)
        norm_tensor = torch.linalg.vector_norm(gradient.double())
        norm = float(norm_tensor)
        if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(norm_tensor)) or not norm > 0:
            raise ValueError('Writer did not receive a finite, nonzero gradient')
        result = {'loss': float(loss.detach()), 'writer_gradient_l2': norm,
                  'writer_gradient_max_abs': float(gradient.abs().max()),
                  'gradient_l2_per_18d_slot': [float(torch.linalg.vector_norm(chunk.double()))
                                             for chunk in gradient.split(18, dim=0)],
                  'active_slots': slot_mask[0].tolist(),
                  'writer_gradient_nonzero': int(torch.count_nonzero(gradient)),
                  'writer_parameters': gradient.numel(),
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'memory_fact': expected, 'teacher_forced_prefix': tokenizer.decode(prefix[0]),
                  'cross_attention_enabled': False,
                  'limitations': ['One artificial fact, fixed address and teacher-forced prefix.',
                                  'Writer initialized to reconstruct this case, not trained or evaluated for generalization.',
                                  'Slot mask derives from the supplied initial packet; not a learned allocation policy.',
                                  'Native reader and vocabulary lookup still detach; gradient is through other research adapters.',
                                  'No optimizer step, natural routing, persistence, or deployed change.']}
        (audit.OUT / 'gradient.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        done = True
        print(json.dumps({'writer_gradient_l2': norm}), flush=True)
    finally:
        for module, enabled in zip(cross_modules, original_flags):
            module.enabled = enabled


audit.PROBE_OBSERVER = observe
if __name__ == '__main__':
    audit.main()
    if not done:
        raise ValueError('Gradient probe was not executed')
