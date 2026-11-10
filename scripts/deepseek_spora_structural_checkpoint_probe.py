"""Round-trip real SPORA replacement layers and individual state in a new process."""
import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_memory_answer_distill import SOURCE_CHECKPOINT, _load_candidate, _memory_args
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import load_integrated, save_integrated
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

PAIR = ROOT/'runs/deepseek-spora-diverse-memory-replay-20261111'
OUTPUT = ROOT/'runs/deepseek-spora-structural-checkpoint-20261111'
CHECKPOINT = OUTPUT/'individual-with-replacements.pt'
ROWS = ({'question': '小林现在喜欢什么饮品？', 'key': 6, 'intent': 'current', 'event_time': 1, 'expected': '水'},
        {'question': '小林以前喜欢什么饮品？', 'key': 6, 'intent': 'past', 'event_time': 0, 'expected': '咖啡'})


def base_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    return base, tokenizer


def measure(integrated, organism, state, tokenizer):
    results = []
    with torch.inference_mode():
        for row in ROWS:
            inputs = closed_think_inputs(tokenizer, row['question'], 'cuda')
            generated = integrated.generate(**inputs, **_memory_args(organism, state, row),
                                            max_new_tokens=8, do_sample=False,
                                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            tail = generated[0, inputs['input_ids'].shape[1]:].tolist()
            answer = tokenizer.decode(tail, skip_special_tokens=True).strip()
            results.append({'question': row['question'], 'intent': row['intent'], 'expected': row['expected'],
                            'answer': answer, 'token_ids': tail,
                            'exact_correct': answer in (row['expected'], row['expected']+'.', row['expected']+'。')})
    return {'rows': results, 'state_digest': frozen_digest(organism, state)}


def verify(checkpoint):
    torch.set_num_threads(2)
    base, tokenizer = base_and_tokenizer()
    integrated, organism, state = load_integrated(checkpoint, base, MODEL, individual_device='cuda')
    result = measure(integrated, organism, state, tokenizer)
    result['checkpoint_sha256'] = sha(checkpoint)
    result['replacement_lengths'] = {str(i): integrated.base.model.layers[i].max_length for i in (6, 12)}
    return result


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base weight mismatch')
    training = json.loads((PAIR/'result.json').read_text(encoding='utf-8'))
    for i in (6, 12):
        if sha(PAIR/f'layer-{i}.pt') != training['layers'][str(i)]['candidate_sha256']:
            raise ValueError('Candidate checkpoint mismatch')
    torch.manual_seed(22850)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    protected = {'base': sha(MODEL/'model.safetensors'), 'source': sha(SOURCE_CHECKPOINT),
                 'pair': {str(i): sha(PAIR/f'layer-{i}.pt') for i in (6, 12)}}
    base, tokenizer = base_and_tokenizer()
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    state = fresh_controlled_state(organism)
    for value in (1, 6):
        state, decision = observe_sourced(organism, state, 6, value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled write rejected')
    blocks = {i: _load_candidate(PAIR/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    for i, block in blocks.items():
        base.model.layers[i] = block
    before = measure(integrated, organism, state, tokenizer)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    save_integrated(CHECKPOINT, organism, state, integrated.reader, MODEL, integrated.layer_indices, replacements=blocks)
    checkpoint_hash = sha(CHECKPOINT)
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    if payload['metadata']['format'] != 'spora-integrated-open-base-v3':
        raise ValueError('Structural checkpoint format not recorded')
    metadata = payload['metadata']
    del payload, integrated, organism, state, base, tokenizer, blocks, block
    gc.collect()
    torch.cuda.empty_cache()
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--verify', str(CHECKPOINT)],
                           capture_output=True, text=True, encoding='utf-8', timeout=360, check=True)
    after = json.loads(child.stdout.strip().splitlines()[-1])
    if after['checkpoint_sha256'] != checkpoint_hash or after['state_digest'] != before['state_digest']:
        raise ValueError('Checkpoint or state differs after reload')
    same_tokens = [a['token_ids'] == b['token_ids'] for a, b in zip(before['rows'], after['rows'])]
    report = {'before': before, 'after_independent_process': after, 'same_tokens': same_tokens,
              'structural_format': metadata['format'], 'feature_variant_sha256': metadata['feature_variant_sha256'],
              'replacement_specs': metadata['replacements'], 'checkpoint_sha256': checkpoint_hash,
              'checkpoint_bytes': CHECKPOINT.stat().st_size, 'protected_sha256': protected,
              'elapsed_seconds': time.perf_counter()-started,
              'limitations': ['Synthetic state with one explicitly addressed entity and two controlled facts.',
                              'No automatic language routing; caller provides addresses and time slots.',
                              'Independent reload is not a crash recovery or multi-user test.',
                              'The large frozen base weights and tokenizer remain separate files.',
                              'Finite short-answer samples do not establish open-chat quality.']}
    if sha(MODEL/'model.safetensors') != protected['base'] or sha(SOURCE_CHECKPOINT) != protected['source'] or any(
        sha(PAIR/f'layer-{i}.pt') != protected['pair'][str(i)] for i in (6, 12)
    ):
        raise ValueError('Protected source changed')
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'same_tokens': same_tokens, 'correct_before': sum(r['exact_correct'] for r in before['rows']),
                      'correct_after': sum(r['exact_correct'] for r in after['rows'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(Path(args.verify)), ensure_ascii=True), flush=True)
    else:
        run()
