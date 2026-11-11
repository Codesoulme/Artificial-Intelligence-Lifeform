"""Compare old and replay-refined pair on the prior fixed memory probe."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_fresh_entity_graft_eval import FACTS, TEMPLATES
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import SOURCE_CHECKPOINT, _load_candidate, _memory_args
from scripts.deepseek_spora_pair_general_replay import OUTPUT as CANDIDATE, PAIR
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import grow
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-pair-replay-memory-eval-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    info = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    for i in (6, 12):
        if (sha(CANDIDATE / f'layer-{i}.pt') != info['candidate_sha256'][str(i)]
                or sha(PAIR / f'layer-{i}.pt') != info['source_pair_sha256'][str(i)]):
            raise ValueError('Pair digest mismatch')
    source_sha = sha(SOURCE_CHECKPOINT)
    if source_sha != info['source_checkpoint_sha256']:
        raise ValueError('Individual checkpoint mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed(skip_replaced=(6, 12)).eval()
    integrated, organism, original = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    original_digest = frozen_digest(organism, original)
    state = fresh_controlled_state(organism)
    rows = []
    for index, (name, key, old, new) in enumerate(FACTS):
        for value in (old, new):
            if state.cursor >= len(state.pages) * organism.config.time_count:
                state = grow(organism, state)
            state, decision = observe_sourced(organism, state, key, value, 'direct', 1.0)
            if not decision['current_write']:
                raise ValueError('Controlled write rejected')
        for intent, template in TEMPLATES.items():
            question = template.format(name=name)
            rows.append({'name': name, 'key': key, 'intent': intent, 'question': question,
                         'event_time': 2 * index + int(intent != 'past'),
                         'value': old if intent == 'past' else new,
                         'other_value': new if intent == 'past' else old,
                         'prefix': closed_think_inputs(tokenizer, question, 'cuda')})
    state_digest = frozen_digest(organism, state)
    results = {}
    for mode, directory in (('before', PAIR), ('after', CANDIDATE)):
        for i in (6, 12):
            base.model.layers[i] = _load_candidate(directory / f'layer-{i}.pt', i, base).eval()
        if any(p.is_meta for p in base.parameters()) or any(b.is_meta for b in base.buffers()):
            raise ValueError('Unmaterialized layer')
        outputs = []
        with torch.inference_mode():
            for row in rows:
                prefix = row['prefix']
                generated = integrated.generate(**prefix, **_memory_args(organism, state, row), max_new_tokens=8,
                                                do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                                eos_token_id=tokenizer.eos_token_id)
                answer = tokenizer.decode(generated[0, prefix['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                expected = WORDS[row['value']]
                outputs.append({'name': row['name'], 'intent': row['intent'], 'question': row['question'],
                                'expected': expected, 'answer': answer,
                                'exact': answer in (expected, expected + '。', expected + '.')})
        results[mode] = {'total': len(outputs), 'exact_correct': sum(r['exact'] for r in outputs), 'rows': outputs}
        print(json.dumps({'mode': mode, 'exact_correct': results[mode]['exact_correct'], 'total': len(outputs)}), flush=True)
    if (sha(SOURCE_CHECKPOINT) != source_sha or frozen_digest(organism, original) != original_digest
            or frozen_digest(organism, state) != state_digest):
        raise ValueError('Protected or temporary individual changed')
    report = {'status': 'pair_replay_fixed_memory_probe', 'results': results, 'source_checkpoint_sha256': source_sha,
              'base_sha256': WEIGHT_SHA, 'source_pair_sha256': info['source_pair_sha256'],
              'candidate_sha256': info['candidate_sha256'], 'seconds': time.perf_counter() - started,
              'limitations': ['Eight previously used authored questions and four finite-value identities.',
                              'Caller supplies memory key and time, no natural-language routing or writing.',
                              'Exact answer match only; small development sample, not independent holdout.',
                              'Research streamed loader and temporary state; no real individual changed.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v['exact_correct'] for k, v in results.items()}), flush=True)


if __name__ == '__main__':
    main()
