"""Compare original-only and paired-consistency answerability probes."""
import copy
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
from scripts.deepseek_joint_answerability_20261127 import cases

OUT = ROOT / 'runs/research/paired-consistency-20261127-v1'
SEED, STEPS, BATCH, LR = 112746, 192, 16, 0.0005


def perturb(prompt):
    for old, new in [('小陈', '小周'), ('小吴', '小林'), ('小邵', '小郑'),
                     ('小安：我们核对一下大家提供的资料。', '主持人：下面整理各人的信息。'),
                     ('小安：现在记下，', '主持人：资料显示，'),
                     ('然后我们再商量别的安排。', '这一条先记录下来。'),
                     ('小安：核对完毕，谢谢大家。', '主持人：这些信息已经整理完成。')]:
        if old.startswith('小安') or old.startswith('然后'):
            if old not in prompt:
                raise ValueError('Expected narration missing')
        prompt = prompt.replace(old, new)
    return prompt


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        path = MODEL / 'model.safetensors'
        before = sha(path)
        if before != WEIGHT_SHA:
            raise ValueError('Wrong base weights')
        original, dev = cases(False), cases(True)
        variants = [{**x, 'prompt': perturb(x['prompt'])} for x in original]
        assert len(original) == len(variants) == 288 and len(dev) == 96
        digest = hashlib.sha256(json.dumps([original, variants, dev], ensure_ascii=False, sort_keys=True).encode('utf8')).hexdigest()
        torch.manual_seed(SEED)
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        vectors = []
        inputs = original + variants + dev
        with torch.no_grad():
            for i, row in enumerate(inputs):
                ids = closed_think_inputs(tok, row['prompt'], 'cuda')['input_ids']
                v = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1].float().cpu()
                if not torch.isfinite(v).all():
                    raise ValueError('Non-finite vector')
                vectors.append(v)
                if (i + 1) % 96 == 0:
                    print(json.dumps({'encoded': i + 1, 'total': len(inputs)}), flush=True)
        del model
        torch.cuda.empty_cache()
        x = torch.stack(vectors)
        y = torch.tensor([r['label'] for r in original], dtype=torch.float32)
        baseline = torch.nn.Linear(x.shape[1], 1)
        paired = copy.deepcopy(baseline)
        optimizers = [torch.optim.AdamW(p.parameters(), lr=LR, weight_decay=0) for p in (baseline, paired)]
        rng = random.Random(SEED)
        trace = []
        for step in range(STEPS):
            indices = rng.sample(range(len(original)), BATCH)
            a, b, target = x[indices], x[[i + len(original) for i in indices]], y[indices]
            log0 = baseline(a).squeeze(-1)
            log1, log2 = paired(a).squeeze(-1), paired(b).squeeze(-1)
            loss0 = F.binary_cross_entropy_with_logits(log0, target)
            loss1 = (F.binary_cross_entropy_with_logits(log1, target) + F.binary_cross_entropy_with_logits(log2, target)) / 2
            loss1 = loss1 + 0.5 * (torch.sigmoid(log1) - torch.sigmoid(log2)).square().mean()
            for optimizer, loss in zip(optimizers, (loss0, loss1)):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            if step % 24 == 0 or step == STEPS - 1:
                trace.append({'step': step + 1, 'baseline_loss': float(loss0.detach()), 'paired_loss': float(loss1.detach())})
        rows = []
        with torch.no_grad():
            for name, probe in [('baseline', baseline), ('paired', paired)]:
                probe.eval()
                probs = torch.sigmoid(probe(x).squeeze(-1)).tolist()
                for i, (row, probability) in enumerate(zip(inputs, probs)):
                    rows.append({'candidate': name, 'split': 'original' if i < 288 else 'variant' if i < 576 else 'development',
                                 'category': row['category'], 'order': row['order'], 'label': row['label'],
                                 'predicted_answerable': probability >= 0.5, 'probability': probability})
                torch.save({'probe': probe.state_dict(), 'base_sha256': before, 'seed': SEED,
                            'dataset_sha256': digest}, OUT / (name + '.pt'))
        after = sha(path)
        result = {'rows': rows, 'trace': trace, 'seed': SEED, 'steps': STEPS, 'batch': BATCH, 'lr': LR,
                  'threshold': 0.5, 'dataset_sha256': digest, 'base_sha_before': before, 'base_sha_after': after,
                  'candidates': {n: sha(OUT / (n + '.pt')) for n in ('baseline', 'paired')},
                  'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'limitations': ['Synthetic reused development cases, not independent evaluation.',
                                  'Augmentation and consistency changed together; causal effects not isolated.',
                                  'Full context classifier only; no generation or persistence.']}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        if before != after:
            raise ValueError('Base weights changed')
        print(json.dumps({'complete': True, 'seconds': result['seconds']}), flush=True)
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__':
    main()
