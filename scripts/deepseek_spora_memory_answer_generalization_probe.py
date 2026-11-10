"""Test answer-distilled SPORA blocks on unseen temporal wording and held entities."""
import json
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import NAMES
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_memory_answer_distill import (
    LAYERS, OUTPUT as TRAINED_OUTPUT, SOURCE_CHECKPOINT, _evaluate,
    _facts, _load_candidate,
)
from scripts.deepseek_spora_memory_strength_factor_probe import _address
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import load_integrated

OUTPUT = ROOT / 'runs/deepseek-spora-memory-answer-generalization-20261111.json'
TEMPLATES = {
    'current': ('告诉我{name}最常喝的是哪种饮品。', '若只看最新记录，{name}偏爱什么？'),
    'past': ('在那次调整之前，{name}会选什么饮料？', '更改记录之前，{name}通常喝哪种饮品？'),
    'later': ('偏好被调整以后，{name}选择什么饮品？', '最新的记录显示{name}爱喝哪种？'),
}


def run(candidate_dir=TRAINED_OUTPUT, output=OUTPUT, novel_value_pairs=False):
    candidate_dir = Path(candidate_dir)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected DeepSeek base hash mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    value_pairs = ((0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 0), (7, 2))
    state, facts = _facts(organism, fresh_controlled_state(organism),
                          value_pairs if novel_value_pairs else None)
    held_rows = []
    for index in (6, 7):
        fact = facts[index]
        for intent, variants in TEMPLATES.items():
            expected = fact['old_value'] if intent == 'past' else fact['new_value']
            other = fact['new_value'] if intent == 'past' else fact['old_value']
            event_time = fact['old_time'] if intent == 'past' else fact['new_time']
            addresses, slot = _address(organism, fact['key'], intent, index)
            for template in variants:
                question = template.format(name=fact['name'])
                prefix = closed_think_inputs(tokenizer, question, 'cuda')
                answer_ids = tokenizer(WORDS[expected], add_special_tokens=False,
                                       return_tensors='pt').input_ids.to('cuda')
                answer_ids = torch.cat((answer_ids, torch.tensor([[tokenizer.eos_token_id]], device='cuda')), dim=1)
                held_rows.append({'name': fact['name'], 'key': fact['key'], 'intent': intent,
                                  'question': question, 'value': expected, 'other_value': other,
                                  'prefix': prefix, 'answer_ids': answer_ids, 'event_time': event_time,
                                  'variant': 2, 'addresses': addresses, 'slot': slot})
    originals = {layer: base.model.layers[layer] for layer in LAYERS}
    trained_paths = {layer: candidate_dir / f'layer-{layer}.pt' for layer in LAYERS}
    trained = {layer: _load_candidate(path, layer, base) for layer, path in trained_paths.items()}
    baseline = _evaluate(integrated, base, organism, state, originals, originals, tokenizer, held_rows)
    candidate = _evaluate(integrated, base, organism, state, trained, originals, tokenizer, held_rows)
    report = {
        'status': 'memory_answer_distillation_unseen_template_probe',
        'base_weight_sha256': WEIGHT_SHA,
        'individual_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
        'candidate_sha256': {str(layer): sha(path) for layer, path in trained_paths.items()},
        'held_entities': [NAMES[6], NAMES[7]],
        'novel_held_value_pairs': novel_value_pairs,
        'held_templates': TEMPLATES,
        'baseline': baseline,
        'candidate': candidate,
        'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'device': torch.cuda.get_device_name(0),
        'limitations': [
            'Two held entities and six unseen question templates.',
            'For the novel-value split, both directed old-to-new pairs are absent from training; the eight-value vocabulary itself is shared.',
            'Exact-value scoring can reject a correct paraphrase or a response exceeding eight generated tokens.',
            'Development probe only; it does not test open-ended conversation or autonomous routing.',
        ],
    }
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'baseline': baseline['accuracy'], 'candidate': candidate['accuracy']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-dir', type=Path, default=TRAINED_OUTPUT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--novel-value-pairs', action='store_true')
    args = parser.parse_args()
    run(args.candidate_dir, args.output, args.novel_value_pairs)
