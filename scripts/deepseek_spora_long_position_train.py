"""Train only new position vectors while retaining all prior candidate weights."""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from scripts.deepseek_spora_integrated_cache_dialogue_probe import POST
from scripts.deepseek_spora_long_position_probe import CASES, answer_metrics

OUTPUT = ROOT/'runs/deepseek-spora-long-position-train-20261111'
STEPS = 36
SEED = 18701
TRAIN = (
    ('小王计划在假期照顾家里的植物。窗台上有一盆绿萝和一盆仙人掌，阳台上有两盆月季。'
     '绿萝喜欢明亮但不暴晒的环境，仙人掌不需要频繁浇水。月季最近长出了新的叶子。'
     '小王先查看了土壤的湿度，又把枯叶从花盆里捡出来。下午太阳比较大，她把绿萝移到屋内。'
     '晚饭后她在纸上记录今天做的事情，以便过几天再检查。她还提醒家人不要随意移动花盆。'
     '第二天早上，她准备给需要水的植物浇水，然后把工具收好。'
     '她先检查了土壤，又收拾了枯叶，最后将绿萝搬进屋内。请回答：被搬进屋内的是哪一种植物？', '绿萝'),
    ('老师让大家为社区活动准备书籍。小陈负责整理儿童读物，小李负责搬桌子，小周负责登记。'
     '他们早上九点到达活动中心，先一起打扫房间，然后把桌椅摆放整齐。'
     '中午志愿者送来一批新的图书，大家按照不同类别把它们放到书架。'
     '下午小陈发现少了一张登记表，于是向小周借了一张空白表，重新写下图书的数量。'
     '活动结束后，他们将剩下的文具放进柜子，关好门窗，并把钥匙交给管理员。'
     '老师询问大家的工作安排时，小陈说她整理了读物，小李搬桌子，小周做登记。'
     '根据上面的安排，谁负责搬桌子？', '小李'),
    ('我正在安排一天的工作。早上需要先查看邮件，再整理上午会议所需的材料。'
     '上午十点参加项目讨论，结束后把讨论中确定的任务写下来，并发给团队成员。'
     '中午休息一小时，下午首先检查昨天的实验结果，然后处理一个需要修复的程序问题。'
     '如果问题当天无法解决，就记录已经尝试过的步骤，避免第二天重复工作。'
     '晚上不再安排会议，只花几分钟检查明天的日历，然后关掉电脑休息。'
     '我的同事问我下午的第一项工作是什么，我回答说会先看昨天的实验结果。'
     '按照这份安排，下午首先要做什么？', '检查昨天的实验结果'),
    ('小林和朋友计划去图书馆学习。他们周一讨论过一次，周二又确认了各自的时间。'
     '小林周六上午要参加一个课程，朋友周日下午有家庭聚会，因此他们寻找共同空闲的时间。'
     '图书馆周六开放到晚上八点，周日开放到下午五点，两人都可以乘坐公交车前往。'
     '他们决定周六下午两点在图书馆大厅碰面，一起复习两个小时，然后各自回家。'
     '为了避免忘记，小林把集合时间和地点写在日历上，朋友也把公交路线保存到了手机里。'
     '到了周五晚上，两人再次确认周六的安排没有变化，准备第二天带好书和水。'
     '请问他们约好在哪里见面？', '图书馆大厅'),
)


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    old = json.loads((POST.parent/'result.json').read_text(encoding='utf-8'))
    if sha(POST) != old['candidate_sha256'] or any(
        sha(PAIR/f'layer-{i}.pt') != old['upstream_pair_sha256'][str(i)] for i in (6, 12)
    ):
        raise ValueError('Candidate provenance mismatch')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    replacements = {i: _load_candidate(PAIR/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    replacements[18] = _load_candidate(POST, 18, base).eval()
    original = {i: base.model.layers[i] for i in (6, 12, 18)}
    examples = []
    with torch.inference_mode():
        for question, answer in TRAIN:
            prefix = closed_think_inputs(tokenizer, question, 'cuda')
            length = prefix['input_ids'].shape[1]
            if not 128 < length <= 220:
                raise ValueError(f'Training prefix length outside range: {length}')
            token = tokenizer(answer, add_special_tokens=False, return_tensors='pt').input_ids[:, 0].to('cuda')
            teacher = base(**prefix, use_cache=False).logits[:, -1].float().detach().half().cpu()
            examples.append({'prefix': prefix, 'label': token, 'teacher': teacher,
                             'question': question, 'answer': answer, 'length': length})
    for example in examples:
        example['label'] = example['label'].clone()
    held = [closed_think_inputs(tokenizer, item['prompt'], 'cuda') for item in CASES]
    old_prefix = {}
    for i, block in replacements.items():
        block.extend_positions(256)
        block.requires_grad_(False)
        block.position.weight.requires_grad_(True)
        block.position.weight.register_hook(lambda gradient: torch.cat(
            (torch.zeros_like(gradient[:128]), gradient[128:]), dim=0))
        old_prefix[str(i)] = digest_tensor(block.position.weight[:128])
        base.model.layers[i] = block
    before = [answer_metrics(base, tokenizer, prefix, case['reference']) for case, prefix in zip(CASES, held)]
    optimizer = torch.optim.AdamW([block.position.weight for block in replacements.values()],
                                  lr=0.02, weight_decay=0.0, foreach=False)
    rng = random.Random(SEED)
    trace = []
    for step in range(STEPS):
        example = examples[step % len(examples)] if step < len(examples) else rng.choice(examples)
        logits = base(**example['prefix'], use_cache=False).logits[:, -1].float()
        teacher = example['teacher'].to('cuda', dtype=torch.float32)
        kl = F.kl_div(F.log_softmax(logits/2, dim=-1), F.softmax(teacher/2, dim=-1),
                      reduction='batchmean') * 4
        ce = F.cross_entropy(logits, example['label'])
        loss = kl + ce
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([block.position.weight for block in replacements.values()], 1.0)
        optimizer.step()
        if step in (0, 8, 17, 26, 35):
            trace.append({'step': step+1, 'loss': float(loss.detach()), 'kl': float(kl.detach()),
                          'label_cross_entropy': float(ce.detach())})
            print(json.dumps(trace[-1]), flush=True)
    if any(digest_tensor(block.position.weight[:128]) != old_prefix[str(i)] for i, block in replacements.items()):
        raise ValueError('An existing position vector changed')
    after = [answer_metrics(base, tokenizer, prefix, case['reference']) for case, prefix in zip(CASES, held)]
    OUTPUT.mkdir(parents=False, exist_ok=False)
    tensors = {str(i): block.position.weight.detach().cpu()[128:].clone() for i, block in replacements.items()}
    checkpoint = OUTPUT/'position-tails.pt'
    torch.save({'format': 'spora-position-tail-v1', 'base_weight_sha256': WEIGHT_SHA,
                'sources': {str(i): sha(PAIR/f'layer-{i}.pt') for i in (6, 12)} | {'18': sha(POST)},
                'old_length': 128, 'new_length': 256, 'tails': tensors}, checkpoint)
    report = {'train': [{'question': e['question'], 'answer': e['answer'], 'prompt_tokens': e['length']}
                        for e in examples], 'held_cases': [c['name'] for c in CASES],
              'before': before, 'after': after, 'trace': trace, 'old_position_hashes': old_prefix,
              'position_tail_sha256': sha(checkpoint), 'train_steps': STEPS, 'seed': SEED,
              'elapsed_seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Only four training examples and two previously inspected development prompts.',
                              'Training uses only first answer-token CE and KL, without a memory reader.',
                              'Only positions actually present in training receive gradient.',
                              'No broad long-context or multi-turn acceptance.']}
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'before_nll': [x['reference_mean_nll'] for x in before],
                      'after_nll': [x['reference_mean_nll'] for x in after]}), flush=True)


if __name__ == '__main__':
    run()
