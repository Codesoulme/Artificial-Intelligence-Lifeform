"""Compare reader and output adapter across paired values on original layers."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_spora_sentence_lora_pilot import make_state
from spora.memory_output import MemoryOutputAdapter

SOURCE = ROOT / 'runs/deepseek-spora-memory-output-adapter-v2-20261111'
OUTPUT = ROOT / 'runs/deepseek-spora-original-layer-output-counterfactual-20261114.json'
CASES = (
    (1, '从记录看，阿宁常喝什么饮料？', (2, 7)),
    (3, '从记录看，阿哲常喝什么饮料？', (4, 1)),
    (2, '请查小雨的饮料偏好。', (3, 6)),
    (0, '请告诉我小林现在喜欢哪种饮品？', (1, 7)),
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    candidate_path = SOURCE / 'adapter.pt'
    candidate_sha = p.sha(candidate_path)
    if (base_sha != p.WEIGHT_SHA or candidate_sha !=
            json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))['adapter_sha256']):
        raise ValueError('Protected artifact checksum mismatch')
    candidate = torch.load(candidate_path, map_location='cpu', weights_only=True)
    if candidate['base_sha256'] != base_sha:
        raise ValueError('Candidate base mismatch')
    for layer in (6, 12):
        path = ROOT / 'runs/deepseek-spora-diverse-memory-replay-20261111' / f'layer-{layer}.pt'
        if candidate['candidate_sha256'][str(layer)] != p.sha(path):
            raise ValueError('Candidate training layer provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    with tempfile.TemporaryDirectory(prefix='spora-original-output-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        base = p.load_full_streamed().eval()
        originals = {i: base.model.layers[i] for i in (6, 12)}
        chat = p.runtime.load_chat(base_loader=lambda: base)
        source_digest = p.frozen_digest(chat.organism, chat.state)
        replacements = {i: base.model.layers[i] for i in originals}
        for i in originals:
            base.model.layers[i] = originals[i]
        old_adapter = chat.decoder.output_adapter
        adapter = MemoryOutputAdapter(**candidate['config']).to('cuda').eval()
        adapter.load_state_dict(candidate['state'], strict=True)
        chat.decoder.output_adapter = adapter
        for index, question, values in CASES:
            states = {value: make_state(chat, KEYS[index], value) for value in values}
            digests = {value: p.frozen_digest(chat.organism, state) for value, state in states.items()}
            prefix = closed_think_inputs(chat.tokenizer, question, 'cuda')
            for value in values:
                other = values[1] if value == values[0] else values[0]
                args, evidence = p.runtime._prepare_memory_read(chat.organism, states[value], KEYS[index], 'current', None)
                _, wrong_evidence = p.runtime._prepare_memory_read(chat.organism, states[other], KEYS[index], 'current', None)
                if int(evidence.argmax(-1)[0]) != value or int(wrong_evidence.argmax(-1)[0]) != other:
                    raise ValueError('Controlled memory read did not match written values')
                results = {}
                for label in ('no_memory', 'reader', 'v2', 'v2_swapped'):
                    memory_args = {'use_cache': False} if label == 'no_memory' else args
                    output_evidence = evidence if label == 'v2' else wrong_evidence if label == 'v2_swapped' else None
                    with torch.inference_mode():
                        ids = chat.decoder.generate(**prefix, **memory_args, memory_output_logits=output_evidence,
                                                    max_new_tokens=32, do_sample=False,
                                                    pad_token_id=chat.tokenizer.pad_token_id,
                                                    eos_token_id=chat.tokenizer.eos_token_id)
                    tail = ids[0, prefix['input_ids'].shape[1]:].tolist()
                    answer = chat.tokenizer.decode(tail, skip_special_tokens=True).strip()
                    results[label] = {'answer': answer, 'token_ids': tail,
                                      'ended': chat.tokenizer.eos_token_id in tail,
                                      'exact': answer.rstrip('。.!！ ') == p.WORDS[value],
                                      'contains_target': p.WORDS[value] in answer,
                                      'contains_other': p.WORDS[other] in answer}
                rows.append({'question': question, 'key': KEYS[index], 'target': p.WORDS[value],
                             'target_id': value, 'swapped_evidence': p.WORDS[other], 'results': results})
                print(json.dumps({'done': len(rows), 'target': p.WORDS[value],
                                  'answers': {label: r['answer'] for label, r in results.items()}}, ensure_ascii=True), flush=True)
            if any(p.frozen_digest(chat.organism, state) != digests[value] for value, state in states.items()):
                raise ValueError('Controlled state changed')
        chat.decoder.output_adapter = old_adapter
        for i in originals:
            base.model.layers[i] = replacements[i]
        if p.frozen_digest(chat.organism, chat.state) != source_digest:
            raise ValueError('Original individual changed')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(candidate_path) != candidate_sha:
        raise ValueError('Protected file changed')
    summary = {}
    for label in ('no_memory', 'reader', 'v2', 'v2_swapped'):
        summary[label] = {'cases': len(rows), 'exact': sum(r['results'][label]['exact'] for r in rows),
                          'contains_target': sum(r['results'][label]['contains_target'] for r in rows),
                          'contains_other': sum(r['results'][label]['contains_other'] for r in rows),
                          'ended': sum(r['results'][label]['ended'] for r in rows)}
    result = {'summary': summary, 'rows': rows, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'adapter_sha256': candidate_sha, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Addresses and value writes manually supplied, bypassing actual route and identity.',
                              'Small previously inspected development questions and finite values.',
                              'Output adapter trained with replacement layers, tested here on original layers.',
                              'Deliberately swapped evidence is a diagnostic and not proof of runtime mismatch.',
                              'Exact and substring metrics do not establish full semantic correctness.',
                              'No temporal, unknown-memory, multi-turn, or persistent-user evaluation.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
