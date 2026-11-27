"""Frozen backbone: full-dialogue vs question-only answerability probes."""
import copy
import hashlib
import json
import random
import re
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha

SOURCE = ROOT / 'runs/research/molweni-source-20261127-v1/MRC(withDiscourse)/train.json'
SOURCE_SHA = 'dc9c374d85e1174188e13ea5dafc15ab7551ad7f234203996a5eb4683edbb8ac'
OUT = ROOT / 'runs/research/molweni-context-probe-20261127-v1'
SEED, STEPS, BATCH, LR = 112746, 192, 16, 0.0005


def select():
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError('Unexpected Molweni training source')
    data = json.loads(SOURCE.read_text(encoding='utf8'))['data']
    if data['title'] != 'train':
        raise ValueError('Unexpected split')
    buckets = {(split, label): [] for split in ('train', 'development') for label in (0, 1)}
    for d in data['dialogues']:
        context = d['context']
        group = hashlib.sha256(context.encode('utf8')).hexdigest()
        split = 'development' if int(group[:4], 16) < 13107 else 'train'
        speakers = {x['speaker'] for x in d['edus'] if x['speaker']}
        for qa in d['qas']:
            label = int(not qa['is_impossible'])
            named = any(re.search(r'(?<!\w)' + re.escape(s) + r'(?!\w)', qa['question'], re.I) for s in speakers)
            buckets[(split, label)].append({'split': split, 'label': label, 'context': context,
                 'question': qa['question'], 'id': qa['id'], 'group': group, 'speaker_name_in_question': named})
    rng = random.Random(SEED)
    selected = []
    for split, n in [('train', 128), ('development', 48)]:
        for label in (0, 1):
            bucket = buckets[(split, label)]
            if len(bucket) < n:
                raise ValueError('Insufficient examples')
            selected.extend(rng.sample(bucket, n))
    assert not ({x['group'] for x in selected if x['split'] == 'train'} &
                {x['group'] for x in selected if x['split'] == 'development'})
    return selected


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        before = sha(MODEL / 'model.safetensors')
        if before != WEIGHT_SHA:
            raise ValueError('Unexpected base')
        items = select()
        signature = hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True).encode('utf8')).hexdigest()
        torch.manual_seed(SEED)
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        vectors = {'context': [], 'question': []}
        with torch.no_grad():
            for mode in vectors:
                for i, item in enumerate(items):
                    prompt = ('Conversation:\n' + item['context'] + '\nQuestion: ' if mode == 'context' else 'Question: ') + item['question']
                    ids = closed_think_inputs(tok, prompt, 'cuda')['input_ids']
                    v = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1].float().cpu()
                    if not torch.isfinite(v).all():
                        raise ValueError('Non-finite vector')
                    vectors[mode].append(v)
                    if (i + 1) % 88 == 0:
                        print(json.dumps({'mode': mode, 'encoded': i + 1, 'total': len(items)}), flush=True)
        del model
        torch.cuda.empty_cache()
        x = {k: torch.stack(v) for k, v in vectors.items()}
        labels = torch.tensor([v['label'] for v in items], dtype=torch.float32)
        train = [i for i, v in enumerate(items) if v['split'] == 'train']
        probe_context = torch.nn.Linear(x['context'].shape[1], 1)
        probes = {'context': probe_context, 'question': copy.deepcopy(probe_context)}
        optimizers = {k: torch.optim.AdamW(v.parameters(), lr=LR, weight_decay=0) for k, v in probes.items()}
        rng = random.Random(SEED)
        trace = []
        for step in range(STEPS):
            indices = rng.sample(train, BATCH)
            losses = {}
            for mode, probe in probes.items():
                loss = F.binary_cross_entropy_with_logits(probe(x[mode][indices]).squeeze(-1), labels[indices])
                optimizers[mode].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[mode].step()
                losses[mode] = float(loss.detach())
            if step % 24 == 0 or step == STEPS - 1:
                trace.append({'step': step + 1, **losses})
        rows = []
        with torch.no_grad():
            for mode, probe in probes.items():
                probs = torch.sigmoid(probe(x[mode]).squeeze(-1)).tolist()
                for i, (item, probability) in enumerate(zip(items, probs)):
                    rows.append({'index': i, 'mode': mode, 'split': item['split'], 'label': item['label'],
                                 'speaker_name_in_question': item['speaker_name_in_question'],
                                 'probability': probability, 'predicted_answerable': probability >= 0.5})
                torch.save({'probe': probe.state_dict(), 'base_sha256': before, 'source_sha256': SOURCE_SHA,
                            'selection_sha256': signature, 'seed': SEED}, OUT / (mode + '.pt'))
        after = sha(MODEL / 'model.safetensors')
        result = {'rows': rows, 'trace': trace, 'seed': SEED, 'steps': STEPS, 'batch': BATCH, 'lr': LR,
                  'threshold': 0.5, 'selection_sha256': signature, 'source_sha256': SOURCE_SHA,
                  'base_sha_before': before, 'base_sha_after': after,
                  'candidates': {n: sha(OUT / (n + '.pt')) for n in probes},
                  'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'limitations': ['Small class-balanced internal split, not natural prevalence or official validation.',
                                  'English dialogue, probe classification only; no generation or persistent memory.',
                                  'Literal speaker-name match is not validated entity attribution.']}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        if after != before:
            raise ValueError('Base changed')
        print(json.dumps({'complete': True, 'seconds': result['seconds']}), flush=True)
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__':
    main()
