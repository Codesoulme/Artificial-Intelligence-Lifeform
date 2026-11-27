"""Measure frozen probe sensitivity to replacing the context of the same question."""
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_molweni_context_probe_20261127 import select, SOURCE, SOURCE_SHA

PARENT = ROOT / 'runs/research/molweni-context-probe-20261127-v1'
OUT = ROOT / 'runs/research/molweni-context-intervention-20261127-v1'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        paths = {'base': MODEL / 'model.safetensors', 'source': SOURCE,
                 'probe': PARENT / 'context.pt', 'report': PARENT / 'result.json'}
        before = {k: sha(v) for k, v in paths.items()}
        parent = json.loads(paths['report'].read_text(encoding='utf8'))
        if before['base'] != WEIGHT_SHA or before['source'] != SOURCE_SHA or before['probe'] != parent['candidates']['context']:
            raise ValueError('Unexpected protected inputs')
        items = select()
        signature = hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True).encode('utf8')).hexdigest()
        if signature != parent['selection_sha256']:
            raise ValueError('Selection differs from source experiment')
        dev = [(i, item) for i, item in enumerate(items) if item['split'] == 'development']
        inputs = []
        for pos, (idx, item) in enumerate(dev):
            if item['label'] != 1:
                continue
            donor = next(other for _, other in (dev[(pos + step) % len(dev)] for step in range(1, len(dev)))
                         if other['group'] != item['group'])
            for condition, context in [('original', item['context']), ('empty', ''), ('replacement', donor['context'])]:
                inputs.append({'index': idx, 'condition': condition,
                               'prompt': 'Conversation:\n' + context + '\nQuestion: ' + item['question']})
        assert len(inputs) == 144
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        payload = torch.load(paths['probe'], map_location='cpu', weights_only=True)
        if payload['base_sha256'] != WEIGHT_SHA or payload['selection_sha256'] != signature:
            raise ValueError('Candidate provenance mismatch')
        probe = torch.nn.Linear(model.config.hidden_size, 1).eval()
        probe.load_state_dict(payload['probe'])
        rows = []
        with torch.no_grad():
            for i, row in enumerate(inputs):
                ids = closed_think_inputs(tok, row['prompt'], 'cuda')['input_ids']
                v = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1].float().cpu()
                probability = torch.sigmoid(probe(v)).item()
                if not 0 <= probability <= 1:
                    raise ValueError('Nonfinite output')
                rows.append({'index': row['index'], 'condition': row['condition'], 'probability': probability,
                             'predicted_answerable': probability >= 0.5})
                if (i + 1) % 36 == 0:
                    print(json.dumps({'encoded': i + 1, 'total': len(inputs)}), flush=True)
        parent_rows = {r['index']: r for r in parent['rows'] if r['mode'] == 'context' and r['split'] == 'development'}
        deltas = [abs(r['probability'] - parent_rows[r['index']]['probability']) for r in rows if r['condition'] == 'original']
        after = {k: sha(v) for k, v in paths.items()}
        result = {'rows': rows, 'parent_max_probability_delta': max(deltas),
                  'input_hashes': before, 'input_hashes_after': after, 'selection_sha256': signature,
                  'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'limitations': ['Replacement and empty contexts lack independently validated answerability labels.',
                                  'Classification sensitivity is not evidence attribution or persistence.',
                                  'English, small sample, internal development data.']}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        if before != after:
            raise ValueError('Protected input changed')
        print(json.dumps({'complete': True, 'seconds': result['seconds'], 'max_delta': max(deltas)}), flush=True)
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__':
    main()
