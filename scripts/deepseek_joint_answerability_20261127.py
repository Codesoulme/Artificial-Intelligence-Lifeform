"""Joint context/question probe on synthetic rotating-person development cases."""
import hashlib
import json
import random
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
from scripts.deepseek_dialogue_distill_20261127 import REL, statement, question

OUT = ROOT / 'runs/research/joint-answerability-20261127-v1'
SEED, STEPS, LR = 112746, 384, 0.0005


def cases(development):
    rels = list(REL)
    names = ('小夏', '小何', '小唐') if development else ('小陈', '小吴', '小邵')
    result = []
    for i, r1 in enumerate(rels):
        r2, missing = rels[(i + 3) % len(rels)], rels[(i + 5) % len(rels)]
        for variant in range(2 if development else 6):
            people = names[:2] if development else tuple(names[j] for j in range(3) if j != variant // 2)
            absent = names[2] if development else names[variant // 2]
            facts = [(people[0], r1, REL[r1][0]), (people[1], r2, REL[r2][2]),
                     (people[1], r1, REL[r1][1]), (people[0], r2, REL[r2][2])]
            if variant % 2:
                facts.reverse()
            if development:
                text = '记录员：以下是大家刚才分享的情况。\n'
                text += ''.join('记录员：' + statement(*f) + '谢谢你的分享。\n' for f in facts)
                text += '记录员：今天的记录就到这里。\n'
            else:
                text = '小安：我们核对一下大家提供的资料。\n'
                text += ''.join('小安：现在记下，' + statement(*f) + '然后我们再商量别的安排。\n' for f in facts)
                text += '小安：核对完毕，谢谢大家。\n'
            items = [(p, r, 'positive_first' if j == 0 else 'positive_second', 1)
                     for j, p in enumerate(people) for r in (r1, r2)]
            items += [(people[0], missing, 'missing_relation', 0), (absent, r1, 'missing_person', 0)]
            for person, relation, category, label in items:
                prompt = '资料：\n' + text + '\n问题：' + question(person, relation)
                result.append({'prompt': prompt, 'label': label, 'category': category,
                               'order': variant % 2, 'person': person, 'relation': relation,
                               'document': i * (2 if development else 6) + variant})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    def save(name, obj):
        path = OUT / (name + '.tmp')
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        path.replace(OUT / name)
    try:
        before = sha(MODEL / 'model.safetensors')
        if before != WEIGHT_SHA:
            raise ValueError('Unexpected base weights')
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        train, dev = cases(False), cases(True)
        assert len(train) == 288 and len(dev) == 96
        for p in ('小陈', '小吴', '小邵'):
            assert any(x['person'] == p and x['label'] == 1 for x in train)
            assert any(x['person'] == p and x['category'] == 'missing_person' for x in train)
        digest = hashlib.sha256(json.dumps([train, dev], ensure_ascii=False, sort_keys=True).encode('utf8')).hexdigest()
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        vectors = []
        with torch.no_grad():
            for i, row in enumerate(train + dev):
                ids = closed_think_inputs(tok, row['prompt'], 'cuda')['input_ids']
                v = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1].float().cpu()
                if not torch.isfinite(v).all():
                    raise ValueError('Non-finite encoding')
                vectors.append(v)
                if (i + 1) % 48 == 0:
                    print(json.dumps({'encoded': i + 1, 'total': len(train) + len(dev)}), flush=True)
        del model
        torch.cuda.empty_cache()
        x = torch.stack(vectors)
        probe = torch.nn.Linear(x.shape[1], 1)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=0)
        rng = random.Random(SEED)
        order = list(range(len(train)))
        trace = []
        for step in range(STEPS):
            if step % len(order) == 0:
                rng.shuffle(order)
            idx = order[step % len(order)]
            loss = F.binary_cross_entropy_with_logits(probe(x[idx]).view(()), torch.tensor(float(train[idx]['label'])))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 48 == 0 or step == STEPS - 1:
                trace.append({'step': step + 1, 'loss': float(loss.detach())})
        probe.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(probe(x).squeeze(-1)).tolist()
        rows = []
        for i, (row, prob) in enumerate(zip(train + dev, probabilities)):
            rows.append({k: v for k, v in row.items() if k != 'prompt'} | {
                'split': 'train' if i < len(train) else 'development', 'probability': prob,
                'predicted_answerable': prob >= 0.5})
        torch.save({'probe': probe.state_dict(), 'base_sha256': before, 'seed': SEED,
                    'dataset_sha256': digest}, OUT / 'candidate.pt')
        after = sha(MODEL / 'model.safetensors')
        save('result.json', {'rows': rows, 'trace': trace, 'seed': SEED, 'steps': STEPS, 'lr': LR,
             'threshold': 0.5, 'dataset_sha256': digest, 'base_sha_before': before, 'base_sha_after': after,
             'candidate_sha256': sha(OUT / 'candidate.pt'), 'seconds': time.perf_counter() - start,
             'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
             'limitations': ['Synthetic development data; overlapping relations, values and templates.',
                             'Full context required; no generation, persistence or independent test.',
                             'Linear probe is separate from frozen backbone.']})
        if after != before:
            raise ValueError('Base weights changed')
        print(json.dumps({'complete': True, 'seconds': time.perf_counter() - start}), flush=True)
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__':
    main()
