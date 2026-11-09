"""Local Qwen baseline; no external answer, memory prompts or training."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from spora.growing import load
from spora.language_training import frozen_digest

MODEL = ROOT/'runs/open-base-qwen2.5-0.5b-7ae5576'
INDIVIDUAL = ROOT/'checkpoints/individual-prechinese-v3.pt'
OUTPUT = ROOT/'runs/open-base-baseline-20261109.json'
WEIGHT_SHA = 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
INDIVIDUAL_SHA = '6269d7749148c95ed4653928f61642c8d93098defed649baab3f807c71494db9'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
PROMPTS = ('你好，请简单介绍一下你能做什么。', '我今天有点累，你能给我一个放松建议吗？',
           '社会主义核心价值观的个人层面包括哪些内容？')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Model or individual digest mismatch')
    filenames = ('config.json', 'generation_config.json', 'merges.txt', 'tokenizer.json',
                 'tokenizer_config.json', 'vocab.json', 'LICENSE', 'model.safetensors')
    hashes = {name: sha(MODEL/name) for name in filenames}
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
    organism, state, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, state)
    rows = []
    with torch.inference_mode():
        for prompt in PROMPTS:
            formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors='pt').to('cuda')
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = model.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            generated = result[0, inputs.input_ids.shape[1]:]
            rows.append({'prompt': prompt, 'text': tokenizer.decode(generated, skip_special_tokens=True),
                         'generated_tokens': int(generated.numel()), 'seconds': time.perf_counter()-start})
    if (before != frozen_digest(organism, state) or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or any(sha(MODEL/name) != value for name, value in hashes.items())):
        raise ValueError('Protected input changed')
    report = {'status': 'baseline_only_no_spora_memory_inference', 'model': 'Qwen/Qwen2.5-0.5B-Instruct',
              'revision': REVISION, 'license': 'Apache-2.0', 'model_files_sha256': hashes,
              'individual_sha256': INDIVIDUAL_SHA, 'individual_frozen_digest': before,
              'torch': torch.__version__, 'transformers': transformers.__version__, 'dtype': 'float16',
              'device': torch.cuda.get_device_name(0), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'total_seconds': time.perf_counter()-t0, 'rows': rows,
              'limitations': ['Outputs are from an existing pretrained model, not SPORA language weights.',
                              'No SPORA memory was connected; prompts are not independent evaluation.',
                              'Pretrained behavior and factual reliability are not guaranteed.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
