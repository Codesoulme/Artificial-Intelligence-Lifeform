"""Bounded local DeepSeek-R1-Distill-Qwen-1.5B compatibility baseline."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest

MODEL = ROOT / 'runs/open-base-deepseek-r1-1.5b-ad9f0ae'
OUTPUT = ROOT / 'runs/deepseek-r1-1.5b-baseline-20261109.json'
REVISION = 'ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562'
WEIGHT_SHA = '58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945'
PROMPTS = ('你好，请简单介绍一下你能做什么。', '我今天有点累，你能给我一个放松建议吗？',
           '社会主义核心价值观的个人层面包括哪些内容？',
           '小明有三本书，又买了两本，送出一本。他现在有几本书？请解释。')
FILES = ('config.json', 'generation_config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'LICENSE')


def run(extended=False):
    output_path = OUTPUT.with_name('deepseek-r1-1.5b-extended-baseline-20261109.json') if extended else OUTPUT
    if output_path.exists():
        raise FileExistsError(output_path)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Model or individual digest mismatch')
    hashes = {name: sha(MODEL / name) for name in FILES}
    torch.manual_seed(4602)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    organism, state, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, state)
    rows = []
    with torch.inference_mode():
        for prompt in (PROMPTS[2:3] if extended else PROMPTS):
            formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                                       add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors='pt', add_special_tokens=False).to('cuda')
            torch.cuda.synchronize()
            start = time.perf_counter()
            max_new_tokens = 768 if extended else 192
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6, top_p=0.95,
                                    pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            tokens = output[0, inputs.input_ids.shape[1]:]
            response = tokenizer.decode(tokens, skip_special_tokens=True)
            eos = tokenizer.eos_token_id in tokens.tolist()
            rows.append({'prompt': prompt, 'generated_text': response, 'generated_tokens': int(tokens.numel()),
                         'think_closed': '</think>' in response, 'ended_with_eos': eos,
                         'seconds': time.perf_counter() - start})
            print(json.dumps({'prompt': prompt, 'generated_tokens': int(tokens.numel()),
                              'think_closed': rows[-1]['think_closed'], 'ended_with_eos': eos},
                             ensure_ascii=True), flush=True)
    if (frozen_digest(organism, state) != before or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or any(sha(MODEL / name) != value for name, value in hashes.items())):
        raise ValueError('Protected input changed')
    report = {'status': 'baseline_only_no_spora_memory_inference',
              'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'revision': REVISION,
              'model_files_sha256': hashes, 'individual_sha256': INDIVIDUAL_SHA,
              'torch': torch.__version__, 'transformers': transformers.__version__, 'dtype': 'float16', 'attention': 'sdpa',
              'device': torch.cuda.get_device_name(0), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'total_seconds': time.perf_counter() - started, 'rows': rows,
              'limitations': ['Pretrained baseline, not SPORA language weights or internal memory.',
                              'Hand-selected questions are not an independent chat evaluation.',
                              ('Single question with 768-token limit; answer may be truncated.' if extended else
                               'Generation is sampled and limited to 192 new tokens; answers may be truncated.'),
                              'The eager FP16 attention path produced nonfinite values; this run uses SDPA.',
                              'Source BF16 weights were loaded as FP16 on a GPU without native BF16.']}
    with output_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'total_seconds': report['total_seconds'], 'peak_allocated_bytes': report['peak_allocated_bytes']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended', action='store_true')
    run(parser.parse_args().extended)
