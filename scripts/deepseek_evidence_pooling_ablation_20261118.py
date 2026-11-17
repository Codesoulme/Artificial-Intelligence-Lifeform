"""Controlled pooling/calibration diagnostic. Never writes organism state.

Three fixed seeds, identical heads/data/budget, mean versus last valid token.
Feature extraction happens once and is saved with source fingerprints. Fresh
authored cases are exploratory holdouts, not an independent deployment audit.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts import deepseek_memory_need_hard_train_20261117 as prior
from scripts.memory_need_diverse_data_20261117 import TEST as ORDINARY_DEV
from spora.research_evidence_calibration import calibrate, confusion, rank_auc

DEFAULT_OUT = ROOT / 'runs/deepseek-evidence-pooling-ablation-20261118'
SEEDS = (4171, 4172, 4173)
FRESH = [
    ('阿蓉的快递单号末尾是三二。', '阿蓉今天收到两件快递。', '阿蓓的快递单号末尾是三二。', '阿蓉的快递单号末尾不是三二。', '阿蓉的快递单号末尾是多少？'),
    ('林叔将包裹交给了门卫。', '林叔上午修理了水管。', '刘叔将包裹交给了门卫。', '林叔没有将包裹交给门卫。', '林叔把包裹交给了谁？'),
    ('阿珂的电脑装着蓝松系统。', '阿珂的电脑屏幕很大。', '阿竹的电脑装着蓝松系统。', '阿珂的电脑没有装蓝松系统。', '阿珂的电脑用的是什么系统？'),
    ('小俞的药盒里放着维生素。', '小俞的药盒是圆形的。', '小葛的药盒里放着维生素。', '小俞的药盒里没有维生素。', '小俞的药盒里放着什么？'),
    ('我们的花盆裂开是因为碰撞。', '我们的花盆放在阳台。', '邻居的花盆裂开是因为碰撞。', '我们的花盆不是因为碰撞裂开的。', '我们的花盆为什么裂开？'),
    ('阿玥把文件改名为星河。', '阿玥将文件备份了两次。', '阿棋把文件改名为星河。', '阿玥没有把文件改名为星河。', '阿玥给文件改了什么名字？'),
    ('小彬订了两张票。', '小彬订的是卧铺票。', '小勋订了两张票。', '小彬订的不是两张票。', '小彬订了几张票？'),
    ('阿翎通过电话办理了退货。', '阿翎退货后买了新鞋。', '阿弥通过电话办理了退货。', '阿翎没有通过电话办理退货。', '阿翎通过什么方式办理退货？'),
    ('我家的客房密码是八六一。', '我家的客房里有两张床。', '他家的客房密码是八六一。', '我家的客房密码不是八六一。', '我家的客房密码是什么？'),
    ('阿喆在商场找回了手套。', '阿喆的手套是羊毛的。', '阿芊在商场找回了手套。', '阿喆没有在商场找回手套。', '阿喆在哪里找回了手套？'),
    ('小竞选择退款是因为尺寸不合。', '小竞在周末提交了退款申请。', '小澜选择退款是因为尺寸不合。', '小竞不是因为尺寸不合才退款。', '小竞为什么选择退款？'),
    ('阿旭的实验用了三次才成功。', '阿旭的实验在教室完成。', '阿桐的实验用了三次才成功。', '阿旭的实验不是三次才成功的。', '阿旭的实验用了几次才成功？'),
]


def pooled(hidden, mask):
    if not bool(mask.any(1).all()):
        raise ValueError('Empty token sequence')
    indices = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = indices.masked_fill(mask == 0, -1).max(1).values
    return {'mean': (hidden.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True),
            'last': hidden[torch.arange(len(mask), device=mask.device), last].float()}


def evaluate(rows, logits, threshold, candidate_sets=False):
    labels = [r['need'] for r in rows]
    result = confusion(logits, labels, threshold)
    result['auc'] = rank_auc(logits, labels)
    by_condition = {}
    for row, score in zip(rows, logits):
        entry = by_condition.setdefault(row.get('condition', 'mixed'), {'correct': 0, 'total': 0})
        entry['total'] += 1
        entry['correct'] += int((score >= threshold) == bool(row['need']))
    result['conditions'] = by_condition
    groups = {}
    for row, score in zip(rows, logits):
        if candidate_sets:
            groups.setdefault(row['pair'], []).append((row, score))
    if groups:
        metrics = dict(groups=len(groups), all_correct=0, right_ranked_first=0,
                       accepted_right=0, accepted_wrong_with_right=0, accepted_when_all_wrong=0)
        for group in groups.values():
            if len(group) != 4:
                raise ValueError('Expected four candidates')
            right = next(s for r, s in group if r['need'] == 0)
            wrong = [s for r, s in group if r['need'] == 1]
            metrics['all_correct'] += int(right < threshold and all(s >= threshold for s in wrong))
            metrics['right_ranked_first'] += int(right < min(wrong))
            winner, value = min(group, key=lambda item: item[1])
            metrics['accepted_right'] += int(value < threshold and not winner['need'])
            metrics['accepted_wrong_with_right'] += int(value < threshold and winner['need'])
            metrics['accepted_when_all_wrong'] += int(min(wrong) < threshold)
        result['candidate_sets'] = metrics
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out
    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    torch.set_num_threads(2)
    sources = [Path(__file__), ROOT / 'spora/research_evidence_calibration.py']
    sources += [ROOT / 'scripts' / f for f in (
        'deepseek_memory_need_hard_train_20261117.py', 'memory_need_diverse_data_20261117.py',
        'deepseek_memory_need_context_pairs_20261117.py', 'deepseek_memory_need_hard_context_20261117.py')]
    sources += [MODEL / f for f in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')]
    hashes = {str(p): sha(p) for p in sources}
    if hashes[str(MODEL / 'model.safetensors')] != WEIGHT_SHA:
        raise ValueError('Protected base mismatch')
    splits = {
        'train': prior.make(prior.TRAIN) + prior.hard_make(prior.HARD_TRAIN) + prior.OTHER_TRAIN,
        'calibration': prior.make(prior.CAL) + prior.hard_make(prior.HARD_CAL) + prior.OTHER_CAL,
        'old_hard_dev': prior.hard_make(prior.HARD_TEST),
        'fresh_authored': prior.hard_make(FRESH),
        'mixed_dev': ORDINARY_DEV,
    }
    seen = set()
    for name, rows in splits.items():
        texts = {r['text'] for r in rows}
        if len(texts) != len(rows) or seen & texts:
            raise ValueError('Exact duplicate across or within splits: ' + name)
        seen |= texts
    out.mkdir(parents=True)
    manifest = dict(source_sha256=hashes, seeds=SEEDS, steps=800, learning_rate=.02,
                    policies=['balanced', 'zero_calibration_fp'], splits=splits,
                    limits=['Authored examples, repeated development sets, no population claims.',
                            'Classifier only, not persistent neural memory or deployed control.'])
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    allrows = [r for rows in splits.values() for r in rows]
    features = {'mean': [], 'last': []}
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i in range(0, len(allrows), 8):
            batch = tok([r['text'] for r in allrows[i:i + 8]], padding=True, return_tensors='pt',
                        add_special_tokens=False).to('cuda')
            values = pooled(model.model(**batch, use_cache=False).last_hidden_state, batch['attention_mask'])
            for kind in features:
                features[kind].append(values[kind].cpu())
            if i % 64 == 0:
                print(json.dumps({'feature_rows': min(i + 8, len(allrows)), 'total': len(allrows)}), flush=True)
    matrices = {k: F.normalize(torch.cat(v), dim=-1) for k, v in features.items()}
    torch.save({'features': matrices, 'manifest_sha256': sha(out / 'manifest.json')}, out / 'features.pt')
    peak = torch.cuda.max_memory_allocated()
    del model
    torch.cuda.empty_cache()
    a, b = len(splits['train']), len(splits['train']) + len(splits['calibration'])
    labels = torch.tensor([r['need'] for r in splits['train']], dtype=torch.float32)
    results, states = [], {}
    for kind, x in matrices.items():
        for seed in SEEDS:
            torch.manual_seed(seed)
            gate = nn.Linear(x.shape[-1], 1)
            optimizer = torch.optim.AdamW(gate.parameters(), lr=.02, weight_decay=.001)
            for _ in range(800):
                loss = F.binary_cross_entropy_with_logits(gate(x[:a]).squeeze(-1), labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                scores = gate(x).squeeze(-1).tolist()
            thresholds, _ = calibrate(scores[a:b], [r['need'] for r in splits['calibration']])
            record = {'pooling': kind, 'seed': seed, 'training_loss': float(loss.detach()),
                      'thresholds': thresholds, 'scores': {}, 'metrics': {}}
            offset = 0
            for split, rows in splits.items():
                ss = scores[offset:offset + len(rows)]
                record['scores'][split] = ss
                record['metrics'][split] = {policy: evaluate(rows, ss, t, split in ('old_hard_dev', 'fresh_authored'))
                                           for policy, t in thresholds.items()}
                offset += len(rows)
            states[f'{kind}-{seed}'] = {'state': gate.state_dict(), 'thresholds': thresholds}
            results.append(record)
            print(json.dumps({'pooling': kind, 'seed': seed,
                'fresh': record['metrics']['fresh_authored']['balanced']['candidate_sets']}), flush=True)
    if any(sha(p) != hashes[str(p)] for p in sources):
        raise ValueError('Protected source changed')
    torch.save(states, out / 'heads.pt')
    result = dict(results=results, seconds=time.perf_counter() - started, peak_allocated_bytes=peak,
                  source_unchanged=True, artifacts={f: sha(out / f) for f in ('manifest.json', 'features.pt', 'heads.pt')},
                  versions={'torch': torch.__version__})
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


if __name__ == '__main__':
    main()
