"""Compare progressive grafts with paired, controlled SPORA memory states."""
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
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_layer18_general_long_audit import PAIR
from scripts.deepseek_spora_memory_answer_distill import SOURCE_CHECKPOINT, _load_candidate, _memory_args
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.structure_transfer import SPORALayerBlend

OUTPUT = ROOT / 'runs/deepseek-spora-blend-memory-probe-20261113.json'
QUESTION = '小林现在最喜欢喝什么饮料？'
GENERAL = '一个人迟到了，怎样礼貌地道歉？'
VALUES = (1, 6)  # coffee and water; assigned to the same address in separate states


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Base weight digest mismatch')
    source_sha = sha(SOURCE_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed().eval()
    integrated, organism, original = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    original_digest = frozen_digest(organism, original)
    originals = {i: base.model.layers[i] for i in (6, 12)}
    candidates = {i: _load_candidate(PAIR / f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    states = {}
    for value in VALUES:
        state = fresh_controlled_state(organism)
        state, decision = observe_sourced(organism, state, KEYS[0], value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled write rejected')
        k = torch.tensor([KEYS[0]], device='cuda')
        retrieved = int(query(organism, state, k, torch.zeros_like(k), torch.zeros_like(k)).argmax(-1)[0])
        if retrieved != value:
            raise ValueError('Controlled native state does not retrieve the expected value')
        states[value] = state
    digests = {value: frozen_digest(organism, state) for value, state in states.items()}
    rows = []
    with torch.inference_mode():
        for strength in (0.0, 0.25, 1.0):
            for i in (6, 12):
                base.model.layers[i] = originals[i] if strength == 0 else SPORALayerBlend(originals[i], candidates[i], strength).eval()
            for condition, question, value in (('memory_off', QUESTION, None),
                                               ('coffee', QUESTION, VALUES[0]),
                                               ('water', QUESTION, VALUES[1]),
                                               ('general_off', GENERAL, None),
                                               ('general_unrelated_memory', GENERAL, VALUES[0])):
                inputs = closed_think_inputs(tokenizer, question, 'cuda')
                kwargs = {'use_cache': False} if value is None else _memory_args(
                    organism, states[value], {'key': KEYS[0], 'intent': 'current'})
                generated = integrated.generate(**inputs, **kwargs, max_new_tokens=24, do_sample=False,
                                                pad_token_id=tokenizer.pad_token_id,
                                                eos_token_id=tokenizer.eos_token_id)
                tail = generated[0, inputs['input_ids'].shape[1]:].tolist()
                answer = tokenizer.decode(tail, skip_special_tokens=True).strip()
                rows.append({'strength': strength, 'condition': condition, 'question': question,
                             'expected': None if value is None else WORDS[value],
                             'answer': answer, 'token_ids': tail,
                             'ended': bool(tail and tail[-1] == tokenizer.eos_token_id),
                             'contains_expected': None if value is None else WORDS[value] in answer})
            print(json.dumps({'strength': strength, 'cases': 5}), flush=True)
    if (sha(MODEL / 'model.safetensors') != WEIGHT_SHA or sha(SOURCE_CHECKPOINT) != source_sha
            or frozen_digest(organism, original) != original_digest
            or any(frozen_digest(organism, states[v]) != digests[v] for v in VALUES)):
        raise ValueError('Protected source or controlled state changed')
    report = {'status': 'controlled_blend_memory_development_probe', 'rows': rows,
              'base_sha256': WEIGHT_SHA, 'source_sha256': source_sha,
              'candidate_sha256': {str(i): sha(PAIR / f'layer-{i}.pt') for i in (6, 12)},
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['One authored memory question and one ordinary question; not independent acceptance.',
                              'Program supplies address and creates two counterfactual finite-value states.',
                              'No output adapter bias: isolates native reader but differs from default runtime.',
                              'String containment is not semantic evaluation; 24-token greedy closed-thinking generation.',
                              'No cache in any arm; blended layers execute both branches.']}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
