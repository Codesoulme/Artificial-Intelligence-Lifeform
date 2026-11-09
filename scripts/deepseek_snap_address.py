"""Compare learned nearest-basis addresses with oracle addresses on fixed trajectories."""
import argparse
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

from scripts.deepseek_full_answer import thinking_inputs
from scripts.deepseek_local_baseline import MODEL, REVISION, WEIGHT_SHA
from scripts.deepseek_pooled_address import PooledAddress, extract
from scripts.open_base_addressed_alignment import HELD_PROMPTS
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_real_memory_alignment import native_read
from scripts.open_base_trajectory_comparison import build_trajectories, numeric_correct
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    runs = (ROOT / 'runs').resolve()
    if not source.is_relative_to(runs) or not output.is_relative_to(runs) or output.exists():
        raise ValueError('Expected source and new output under runs')
    source_sha = sha(source)
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != 'deepseek-spora-question-pooled-finite-address-v1'
            or checkpoint.get('weight_sha256') != WEIGHT_SHA or checkpoint.get('individual_sha256') != INDIVIDUAL_SHA
            or checkpoint.get('layers') != [5, 17]):
        raise ValueError('Source provenance mismatch')
    def verify():
        return (sha(MODEL / 'model.safetensors') == WEIGHT_SHA and sha(INDIVIDUAL) == INDIVIDUAL_SHA
                and sha(source) == source_sha)
    if not verify():
        raise ValueError('Protected source mismatch')
    torch.set_num_threads(2)
    torch.manual_seed(4609)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    trajectories = build_trajectories(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    adapter.load_state_dict(checkpoint['readout'], strict=True)
    selector = PooledAddress(model.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    selector.load_state_dict(checkpoint['selector'], strict=True)
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    cases = [(item, key) for item in trajectories for key in item['keys']]
    questions = [HELD_PROMPTS[0].format(key=key) for _, key in cases]
    features, masks = extract(model, tokenizer, questions)
    rows = []
    with torch.inference_mode():
        for index, ((item, key), question) in enumerate(zip(cases, questions)):
            inputs = thinking_inputs(tokenizer, question, 'cuda')
            continuous = selector(features[index:index + 1].to('cuda'), masks[index:index + 1].to('cuda'))
            scores = [continuous[site] @ bases[site].T for site in range(2)]
            predicted = [int(score.argmax(-1)) for score in scores]
            snapped = [bases[site][predicted[site]:predicted[site] + 1] for site in range(2)]
            oracle = [bases[site][key:key + 1] for site in range(2)]
            expected = item['targets'][key]
            row = {'trajectory': item['id'], 'key': key, 'question': question, 'expected': expected,
                   'native': native_read(organism, item['state'], [key])[0], 'predicted_addresses': predicted,
                   'top_cosine': [float(score.max()) for score in scores], 'responses': {}}
            for mode, addresses in (('snapped', snapped), ('oracle', oracle)):
                with memory_hooks([model.model.layers[5], model.model.layers[17]], adapter, item['state'],
                                  current_only=True, addresses=addresses):
                    generated = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                               pad_token_id=tokenizer.eos_token_id)
                raw = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                final = raw.split('</think>', 1)[1].strip() if '</think>' in raw else None
                row['responses'][mode] = {'raw': raw, 'final_answer': final,
                                          'correct': final is not None and numeric_correct(final, expected)}
            rows.append(row)
            if (index + 1) % 4 == 0:
                print(f'checked {index + 1}/{len(cases)}', flush=True)
    if frozen_digest(organism, original) != before or not verify():
        raise ValueError('Protected input changed')
    summary = {'examples': len(rows), 'address_both_correct': sum(r['predicted_addresses'] == [r['key'], r['key']] for r in rows),
               'snapped_correct': sum(r['responses']['snapped']['correct'] for r in rows),
               'oracle_correct': sum(r['responses']['oracle']['correct'] for r in rows),
               'snapped_equals_oracle': sum(r['responses']['snapped']['raw'] == r['responses']['oracle']['raw'] for r in rows)}
    report = {'status': 'nearest_basis_development_only', 'revision': REVISION, 'weight_sha256': WEIGHT_SHA,
              'individual_sha256': INDIVIDUAL_SHA, 'selector_sha256': source_sha, 'summary': summary, 'rows': rows,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['All 64 entity-to-address mappings were supervised; no unseen-entity generalization.',
                              'Oracle address is a diagnostic upper bound and is not used for snapped predictions.',
                              'All trajectories share one original individual and were used in earlier development.',
                              'Fixed short reasoning was supervised; no multi-step reasoning claim.',
                              'Argmax always selects an address; rejection and unknown handling are not established.',
                              'Greedy decoding and exact answer matching are development diagnostics.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
