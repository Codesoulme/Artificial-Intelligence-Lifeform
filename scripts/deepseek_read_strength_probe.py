"""Measure memory-read strength against chat interference and finite recall."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_chat_interference import PROMPTS, SELECTOR
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_trajectory_comparison import numeric_correct
from scripts.spora_unknown_state_probe import CHECKPOINT, OUTPUT as REFERENCE
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks

OUTPUT = ROOT / 'runs/deepseek-read-strength-20261110.json'
STRENGTHS = (0.0, 0.1, 0.25, 0.5, 1.0)


class ScaledReadout(OpenBaseMemory):
    def __init__(self, hidden_width, memory_width, strength):
        super().__init__(hidden_width, memory_width)
        self.strength = strength

    def forward(self, hidden, matrices, site, address=None):
        injected = super().forward(hidden, matrices, site, address)
        return hidden + self.strength * (injected - hidden)


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
    known = [r for r in reference['phases']['interference']['rows'] if r['known_expected']][:8]
    if len(known) != 8:
        raise ValueError('Eight known addresses required')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    adapter = ScaledReadout(model.config.hidden_size, organism.config.memory_width, 1.).to('cuda').eval()
    adapter.load_state_dict(selector['readout'], strict=True)
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    layers = [model.model.layers[5], model.model.layers[17]]
    memory, chat = [], []
    with torch.inference_mode():
        for question in PROMPTS:
            inputs = closed_think_inputs(tokenizer, question, 'cuda')
            baseline = model(**inputs, use_cache=False).logits[0, -1].float()
            baseline_logp = baseline.log_softmax(-1)
            baseline_p = baseline_logp.exp()
            row = {'question': question, 'baseline_top_token': int(baseline.argmax()), 'strengths': {}}
            for strength in STRENGTHS:
                adapter.strength = strength
                address = [basis[known[0]['key']:known[0]['key'] + 1] for basis in bases]
                with memory_hooks(layers, adapter, state, current_only=True, addresses=address):
                    logits = model(**inputs, use_cache=False).logits[0, -1].float()
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError('Nonfinite output logits')
                row['strengths'][str(strength)] = {'same_top_token': int(logits.argmax()) == row['baseline_top_token'],
                    'kl_from_baseline': float((baseline_p * (baseline_logp - logits.log_softmax(-1))).sum())}
            chat.append(row)
            print(f'chat {len(chat)}/{len(PROMPTS)}', flush=True)
        for item in known:
            question = f"对象{item['key']}的值是多少？"
            inputs = closed_think_inputs(tokenizer, question, 'cuda')
            row = {'key': item['key'], 'expected': item['expected'], 'question': question, 'strengths': {}}
            address = [basis[item['key']:item['key'] + 1] for basis in bases]
            for strength in STRENGTHS:
                adapter.strength = strength
                with memory_hooks(layers, adapter, state, current_only=True, addresses=address):
                    generated = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                               pad_token_id=tokenizer.eos_token_id)
                tokens = generated[0, inputs['input_ids'].shape[1]:]
                text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                row['strengths'][str(strength)] = {'text': text, 'correct': numeric_correct(text, item['expected']),
                                                    'ended_with_eos': tokenizer.eos_token_id in tokens.tolist()}
            memory.append(row)
            print(f'memory {len(memory)}/{len(known)}', flush=True)
    if frozen_digest(organism, state) != before or any(sha(p) != digest for p, digest in protected.items()):
        raise ValueError('Protected source changed')
    summary = {str(strength): {'chat_same_first_token': sum(r['strengths'][str(strength)]['same_top_token'] for r in chat),
                               'chat_mean_kl': sum(r['strengths'][str(strength)]['kl_from_baseline'] for r in chat) / len(chat),
                               'memory_correct': sum(r['strengths'][str(strength)]['correct'] for r in memory)}
               for strength in STRENGTHS}
    report = {'status': 'forced_strength_tradeoff_development_only', 'weight_sha256': WEIGHT_SHA,
              'state_sha256': protected[CHECKPOINT], 'selector_sha256': protected[SELECTOR],
              'summary': summary, 'chat': chat, 'memory': memory, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Read is forced; this does not evaluate a learned gate or runtime routing.',
                              'One externally selected known address is reused for all chat prompts.',
                              'Memory queries use the correct externally supplied address, not a learned selector.',
                              'Only four chat prompts and eight known slots from one state.',
                              'Closed thinking, greedy decoding, and 8-token memory answer limit.',
                              'Chat comparison only checks first-token distribution, not full answer quality.',
                              'Scale is applied after FP16 residual rounding; zero strength preserves output exactly.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
