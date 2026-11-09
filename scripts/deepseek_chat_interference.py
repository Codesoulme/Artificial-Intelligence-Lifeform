"""Compare ordinary chat with forced SPORA memory injection and ablations."""
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import CHECKPOINT, OUTPUT as REFERENCE
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks

SELECTOR = ROOT / 'runs/deepseek-question-pooled-3000-20261109/selector.pt'
OUTPUT = ROOT / 'runs/deepseek-chat-interference-20261110.json'
PROMPTS = ('你好，你会说中文吗？', '我今天有点累，请给我一个放松建议。',
           '社会主义核心价值观的个人层面有哪些？', '如何快速提升个人烹饪技巧？')


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, CHECKPOINT, SELECTOR, REFERENCE)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    with REFERENCE.open(encoding='utf-8') as stream:
        reference = json.load(stream)
    if protected[CHECKPOINT] != reference['checkpoint_sha256']:
        raise ValueError('Memory state mismatch')
    selector = torch.load(SELECTOR, map_location='cpu', weights_only=True)
    if (selector.get('format') != 'deepseek-spora-question-pooled-finite-address-v1'
            or selector.get('weight_sha256') != WEIGHT_SHA or selector.get('individual_sha256') != INDIVIDUAL_SHA
            or selector.get('layers') != [5, 17]):
        raise ValueError('Adapter provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, state, _ = load(CHECKPOINT, 'cpu')
    before = frozen_digest(organism, state)
    if before != reference['checkpoint_frozen_digest']:
        raise ValueError('State digest mismatch')
    known_keys = [r['key'] for r in reference['phases']['interference']['rows'] if r['known_expected']][:2]
    if len(known_keys) != 2:
        raise ValueError('Two known addresses required')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    adapter.load_state_dict(selector['readout'], strict=True)
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    empty = SimpleNamespace(current=[torch.zeros_like(tensor) for tensor in state.current], pages=[])
    layers = [model.model.layers[5], model.model.layers[17]]
    rows = []
    with torch.inference_mode():
        for question in PROMPTS:
            inputs = closed_think_inputs(tokenizer, question, 'cuda')
            variants = {}
            for name, active_state, key in (('baseline', None, None), ('zero_matrices', empty, known_keys[0]),
                                            ('memory_a', state, known_keys[0]), ('memory_b', state, known_keys[1])):
                addresses = None if key is None else [basis[key:key + 1] for basis in bases]
                context = (nullcontext() if active_state is None else
                           memory_hooks(layers, adapter, active_state, current_only=True, addresses=addresses))
                with context:
                    logits = model(**inputs, use_cache=False).logits[0, -1].float()
                    if not bool(torch.isfinite(logits).all()):
                        raise ValueError('Nonfinite output logits')
                    generated = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                               pad_token_id=tokenizer.eos_token_id)
                tokens = generated[0, inputs['input_ids'].shape[1]:]
                variants[name] = {'tokens': tokens.cpu().tolist(),
                                  'text': tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                                  'ended_with_eos': tokenizer.eos_token_id in tokens.tolist(),
                                  'top_token': int(logits.argmax()), 'logits': logits.cpu()}
            baseline = variants['baseline']
            base_logp = baseline['logits'].log_softmax(-1)
            base_p = base_logp.exp()
            for name, item in variants.items():
                item['same_tokens_as_baseline'] = item['tokens'] == baseline['tokens']
                item['first_token_kl_from_baseline'] = float((base_p * (base_logp - item['logits'].log_softmax(-1))).sum())
                del item['logits']
            rows.append({'question': question, 'variants': variants})
            print(f'checked {len(rows)}/{len(PROMPTS)}', flush=True)
    if frozen_digest(organism, state) != before or any(sha(p) != digest for p, digest in protected.items()):
        raise ValueError('Protected source changed')
    summary = {name: {'same_output': sum(r['variants'][name]['same_tokens_as_baseline'] for r in rows),
                      'same_first_token': sum(r['variants'][name]['top_token'] == r['variants']['baseline']['top_token']
                                              for r in rows)}
               for name in ('zero_matrices', 'memory_a', 'memory_b')}
    report = {'status': 'forced_memory_interference_development_only', 'weight_sha256': WEIGHT_SHA,
              'state_sha256': protected[CHECKPOINT], 'selector_sha256': protected[SELECTOR],
              'known_keys': known_keys, 'summary': summary, 'rows': rows,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Memory injection is deliberately forced on unrelated prompts; normal router may defer.',
                              'Two known addresses are chosen externally, not predicted from the questions.',
                              'Closed thinking, greedy decoding and 64-token limit are diagnostic and may truncate responses.',
                              'Four hand-selected prompts, one state, one model and one decoding setting.',
                              'Exact output equality is not a general measure of helpfulness or safety.',
                              'Zero matrices are an ablation, not the model initial memory state.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
