"""Trace one matched memory intervention through the frozen model's read path."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch.nn import functional as F
from scripts import deepseek_local_baseline as base
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_joint_writer_reader_pilot import fact_cache
from scripts.deepseek_spora_ttt_shared_joint_pilot import write_sequence
from scripts.deepseek_spora_shared_joint_counterfactual import SOURCE, make_assignments, PEOPLE
from spora.research_ttt_ntp import TTTNTPWriteTargetAdapter, attach_read_adapters, install, use_memory

OUT = ROOT / 'runs/deepseek-spora-shared-signal-diagnostic-20261116.json'


def compare(first, second):
    first, second = first.detach().float().reshape(-1), second.detach().float().reshape(-1)
    return {'first_norm': float(first.norm()), 'second_norm': float(second.norm()),
            'difference_norm': float((first-second).norm()),
            'cosine': float(F.cosine_similarity(first, second, dim=0, eps=1e-12))}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {'base': base.MODEL / 'model.safetensors', 'candidate': SOURCE / 'candidate.pt',
             'report': SOURCE / 'result.json'}
    hashes = {name: base.sha(path) for name, path in paths.items()}
    report = json.loads(paths['report'].read_text(encoding='utf-8'))
    payload = torch.load(paths['candidate'], map_location='cpu', weights_only=True)
    if (hashes['base'] != base.WEIGHT_SHA or hashes['candidate'] != report['candidate_sha256']
            or payload['base_sha256'] != hashes['base']):
        raise ValueError('Source provenance mismatch')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    wrappers = install(model, tuple(payload['layers']))
    readers = attach_read_adapters(wrappers, rank=payload['read_rank'])
    writers = {layer: TTTNTPWriteTargetAdapter(wrappers[layer].base.out_features,
                                                rank=payload['write_rank']).to('cuda')
               for layer in wrappers}
    for layer in wrappers:
        readers[layer].load_state_dict(payload['readers'][str(layer)], strict=True)
        writers[layer].load_state_dict(payload['writers'][str(layer)], strict=True)
        writers[layer].requires_grad_(False)
    model.requires_grad_(False)
    before, after = make_assignments()
    caches = {item: fact_cache(model, tokenizer, wrappers,
                               f'{item[0]}喜欢喝{item[1]}。', item[1])
              for item in set(before + after)}
    states = {label: write_sequence(wrappers, [caches[item] for item in events], writers,
                                    differentiable=False)
              for label, events in (('before', before), ('after', after))}
    prompt = closed_think_inputs(tokenizer, f'请只回答{PEOPLE[0]}喜欢喝什么饮料。', 'cuda')
    observations = {}
    for label, memory in states.items():
        captured = {}
        hooks = []
        for layer, wrapper in wrappers.items():
            def capture(module, args, layer=layer):
                captured[layer] = args[0][0, -1].detach().float()
            hooks.append(wrapper.register_forward_pre_hook(capture))
        try:
            with torch.inference_mode(), use_memory(memory):
                logits = model(**prompt, use_cache=False, return_dict=True).logits[0, -1].float()
        finally:
            for hook in hooks:
                hook.remove()
        reads = {layer: F.linear(captured[layer], memory[layer]) for layer in wrappers}
        with torch.inference_mode():
            adapted = {layer: readers[layer](reads[layer]) for layer in wrappers}
        observations[label] = {'queries': captured, 'reads': reads, 'adapted': adapted,
                               'logits': logits}
    comparisons = {}
    for layer in wrappers:
        comparisons[str(layer)] = {
            'fast_weight': compare(states['before'][layer], states['after'][layer]),
            'query': compare(observations['before']['queries'][layer], observations['after']['queries'][layer]),
            'raw_read': compare(observations['before']['reads'][layer], observations['after']['reads'][layer]),
            'adapter_output': compare(observations['before']['adapted'][layer], observations['after']['adapted'][layer])}
    ids = {value: tokenizer.encode(value, add_special_tokens=False) for value in ('酸奶', '豆浆', '咖啡', '果汁')}
    if any(len(tokens) != 1 for tokens in ids.values()):
        raise ValueError('Diagnostic requires single-token values')
    score = {}
    for label in states:
        logits = observations[label]['logits']
        probs = torch.softmax(logits, dim=-1)
        score[label] = {value: {'token_id': tokens[0], 'logit': float(logits[tokens[0]]),
                                'probability': float(probs[tokens[0]])}
                        for value, tokens in ids.items()}
        score[label]['top_id'] = int(logits.argmax())
    if any(base.sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected input changed')
    result = {'layers': comparisons, 'token_scores': score, 'source_sha256': hashes,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['One known-name, same-question intervention on previously used synthetic facts.',
                              'Read vectors are reconstructed from actual layer inputs and the active state.',
                              'Later-layer inputs depend on earlier memory intervention; this is not a layer ablation.',
                              'Cosine and next-token probabilities cannot establish general recall or causal attribution.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'layers': comparisons, 'token_scores': score}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
