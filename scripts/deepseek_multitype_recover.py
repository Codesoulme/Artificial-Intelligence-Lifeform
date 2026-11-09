"""Evaluate a saved deep-feature gate whose metadata was corrupted by a logging bug."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_guard_adversarial import SECOND_CASES
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_memory_intent import build_cases
from scripts.deepseek_multitype_gate import CLASSES, THRESHOLD, MultiIntentGate, infer, summarize
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_semantic_gate import hide_numbers
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.finite_question_guard import valid_finite_question_form


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    runs = (ROOT / 'runs').resolve()
    if not source.is_relative_to(runs) or not output.is_relative_to(runs) or output.exists():
        raise ValueError('Expected source and new output under runs')
    digest = sha(source)
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != 'deepseek-spora-multitype-intent-v1'
            or checkpoint.get('weight_sha256') != WEIGHT_SHA or checkpoint.get('individual_sha256') != INDIVIDUAL_SHA
            or tuple(checkpoint.get('classes', ())) != CLASSES or checkpoint.get('threshold') != THRESHOLD
            or not isinstance(checkpoint.get('hidden_state_indices'), torch.Tensor)):
        raise ValueError('Unexpected checkpoint or missing known metadata defect')
    def verify():
        return (sha(MODEL / 'model.safetensors') == WEIGHT_SHA and sha(INDIVIDUAL) == INDIVIDUAL_SHA
                and sha(source) == digest)
    if not verify():
        raise ValueError('Protected source mismatch')
    torch.set_num_threads(2)
    held = build_cases(True, augmented=True, malformed=True, contrastive=True)
    stress = [{'category': ('current' if label else category), 'question': question}
              for label, category, question in SECOND_CASES]
    held_texts = sorted({hide_numbers(r['question']) for r in held if valid_finite_question_form(r['question'])})
    stress_texts = sorted({hide_numbers(r['question']) for r in stress if valid_finite_question_form(r['question'])})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    gate = MultiIntentGate(model.config.hidden_size).to('cuda').eval()
    gate.load_state_dict(checkpoint['state'], strict=True)
    hidden_indices = (24, 28)
    held_features, held_masks = extract(model, tokenizer, held_texts, hidden_indices=hidden_indices)
    stress_features, stress_masks = extract(model, tokenizer, stress_texts, hidden_indices=hidden_indices)
    held_lookup = dict(zip(held_texts, infer(gate, held_features, held_masks)))
    stress_lookup = dict(zip(stress_texts, infer(gate, stress_features, stress_masks)))
    results = {}
    for name, cases, lookup in (('held', held, held_lookup), ('stress_development', stress, stress_lookup)):
        rows = []
        for case in cases:
            text = hide_numbers(case['question'])
            form_ok = valid_finite_question_form(case['question'])
            probs = lookup[text] if form_ok else None
            predicted = CLASSES[max(range(len(CLASSES)), key=lambda i: probs[i])] if probs is not None else None
            accepted = bool(form_ok and predicted == 'current' and probs[0] >= THRESHOLD)
            rows.append(dict(case, sanitized=text, form_ok=form_ok, predicted_type=predicted,
                             current_probability=probs[0] if probs is not None else None, accepted=accepted))
        results[name] = summarize(rows)
    if not verify():
        raise ValueError('Protected input changed')
    recovered = {'status': 'multitype_deep_feature_recovered_evaluation', 'source_sha256': digest,
                 'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                 'hidden_state_indices': hidden_indices, 'results': results,
                 'limitations': ['Original checkpoint metadata contains a tensor due to a variable-shadowing bug.',
                                 'Hidden indices 24 and 28 come from the recorded training invocation, not checkpoint metadata.',
                                 'No retraining; this re-evaluates the saved gate.',
                                 'Few synthetic templates; stress cases are development, not final acceptance.',
                                 'No end-to-end memory or open-domain reliability established.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(recovered, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({name: {k: v for k, v in value.items() if k != 'rows'} for name, value in results.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
