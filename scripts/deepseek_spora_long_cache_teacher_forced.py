"""Compare cached and full forward logits across extended positions."""
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.language_training import frozen_digest
from spora.structure_transfer import SPORALayerReplacement

OUTPUT = ROOT / 'runs/deepseek-spora-long-cache-teacher-forced-20261112.json'
CHECKPOINTS = (32, 128, 192, 256, 384, 512)
TEXT = ('清晨的街道逐渐热闹起来，行人从路口经过。'
        '图书馆的窗边放着几本书，读者正在安静阅读。'
        '午后的天气有些变化，树叶在微风中轻轻摇动。'
        '朋友约好周末散步，沿途可以看到一座小桥。'
        '厨房里正在准备晚餐，桌上摆着蔬菜和水果。'
        '工作人员整理资料，将需要查看的内容按顺序放好。'
        '一位学生在纸上记录问题，准备稍后向老师请教。'
        '太阳落山之后，路灯亮起，附近的商店陆续关门。')


def compare_logits(cached, full):
    c, f = cached.detach().float(), full.detach().float()
    if c.shape != f.shape or c.ndim != 1 or not bool(torch.isfinite(c).all() and torch.isfinite(f).all()):
        raise ValueError('Invalid comparison logits')
    delta = (c - f).abs()
    kl = F.kl_div(F.log_softmax(c, dim=-1), F.softmax(f, dim=-1), reduction='sum')
    return {'top_match': int(c.argmax()) == int(f.argmax()), 'cached_top': int(c.argmax()),
            'full_top': int(f.argmax()), 'max_absolute_difference': float(delta.max()),
            'mean_absolute_difference': float(delta.mean()), 'kl_full_to_cached': float(kl)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = sha(runtime.CHECKPOINT)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    raw = tokenizer(TEXT * 20, add_special_tokens=False, return_tensors='pt').input_ids
    if raw.shape[1] < CHECKPOINTS[-1]:
        raise ValueError('Synthetic input too short')
    ids = raw[:, :CHECKPOINTS[-1]].to('cuda')
    token_digest = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        base = chat.base.eval()
        if any(p.is_meta for p in base.parameters()) or any(b.is_meta for b in base.buffers()):
            raise ValueError('Unmaterialized layer')
        layers = {i: x for i, x in enumerate(base.model.layers) if isinstance(x, SPORALayerReplacement)}
        if set(layers) != {6, 12} or any(x.max_length < CHECKPOINTS[-1] for x in layers.values()):
            raise ValueError('Unexpected replacement lengths')
        digest = frozen_digest(chat.organism, chat.state)

        def record(length, cached_logits):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                full = base(input_ids=ids[:, :length], attention_mask=torch.ones_like(ids[:, :length]),
                            use_cache=False).logits[0, -1].float()
            torch.cuda.synchronize()
            metrics = compare_logits(cached_logits, full)
            rows.append({'length': length, **metrics, 'full_seconds': time.perf_counter() - start})
            print(json.dumps(rows[-1]), flush=True)

        with torch.inference_mode():
            first = CHECKPOINTS[0]
            output = base(input_ids=ids[:, :first], attention_mask=torch.ones_like(ids[:, :first]), use_cache=True)
            cache = output.past_key_values
            if cache is None:
                raise ValueError('Model returned no mutable cache')
            last = output.logits[0, -1].float()
            del output
            record(first, last)
            for index in range(first, CHECKPOINTS[-1]):
                output = base(input_ids=ids[:, index:index + 1], attention_mask=torch.ones_like(ids[:, :index + 1]),
                              past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                length = index + 1
                if length in CHECKPOINTS:
                    last = output.logits[0, -1].float()
                    record(length, last)
                if length % 64 == 0:
                    print(json.dumps({'processed_tokens': length}), flush=True)
                del output
        if frozen_digest(chat.organism, chat.state) != digest:
            raise ValueError('Protected individual changed')
    if sha(runtime.CHECKPOINT) != source_sha or sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected source changed')
    summary = {'checkpoints': len(rows), 'top_matches': sum(r['top_match'] for r in rows),
               'largest_logit_difference': max(r['max_absolute_difference'] for r in rows),
               'max_kl': max(r['kl_full_to_cached'] for r in rows)}
    report = {'status': 'teacher_forced_extended_cache_audit', 'summary': summary, 'rows': rows,
              'input_token_sha256': token_digest, 'source_checkpoint_sha256': source_sha,
              'base_sha256': WEIGHT_SHA, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['One repeated synthetic Chinese passage, fixed teacher-forced token sequence.',
                              'Only six position checkpoints; no greedy generation beyond these positions checked.',
                              'Cache and full outputs use same model, no correctness or long-context comprehension guarantee.',
                              'No memory reader, multi-turn chat, individual mutation or crash recovery.',
                              'Research streamed loader; floating point effects can differ by device and runtime.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
