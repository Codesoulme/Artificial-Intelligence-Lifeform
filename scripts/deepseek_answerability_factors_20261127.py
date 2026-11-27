"""Paired name and framing interventions for the frozen answerability probe."""
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
from scripts.deepseek_joint_answerability_20261127 import cases

SOURCE = ROOT / 'runs/research/joint-answerability-20261127-v1'
OUT = ROOT / 'runs/research/answerability-factors-20261127-v1'


def transform(prompt, new_names, new_framing):
    if new_names:
        for old, new in [('小陈', '小唐'), ('小吴', '小夏'), ('小邵', '小何')]:
            prompt = prompt.replace(old, new)
    if new_framing:
        for old, new in [('小安：我们核对一下大家提供的资料。', '记录员：以下是大家刚才分享的情况。'),
                         ('小安：现在记下，', '记录员：'),
                         ('然后我们再商量别的安排。', '谢谢你的分享。'),
                         ('小安：核对完毕，谢谢大家。', '记录员：今天的记录就到这里。')]:
            if old not in prompt:
                raise ValueError('Expected framing text absent')
            prompt = prompt.replace(old, new)
    return prompt


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        paths = {'base': MODEL / 'model.safetensors', 'candidate': SOURCE / 'candidate.pt', 'report': SOURCE / 'result.json'}
        before = {k: sha(v) for k, v in paths.items()}
        metadata = json.loads(paths['report'].read_text(encoding='utf8'))
        if before['base'] != WEIGHT_SHA or before['candidate'] != metadata['candidate_sha256']:
            raise ValueError('Unexpected base or candidate')
        original = [x for x in cases(False) if x['document'] % 6 == 0]
        assert len(original) == 48
        inputs = []
        for i, row in enumerate(original):
            for names, framing in [(False, False), (True, False), (False, True), (True, True)]:
                inputs.append({'index': i, 'category': row['category'], 'label': row['label'],
                               'names': names, 'framing': framing,
                               'prompt': transform(row['prompt'], names, framing)})
        dataset_sha = hashlib.sha256(json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode('utf8')).hexdigest()
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        payload = torch.load(paths['candidate'], map_location='cpu', weights_only=True)
        if payload['base_sha256'] != WEIGHT_SHA:
            raise ValueError('Candidate uses different base')
        probe = torch.nn.Linear(model.config.hidden_size, 1).eval()
        probe.load_state_dict(payload['probe'])
        rows = []
        with torch.no_grad():
            for i, row in enumerate(inputs):
                ids = closed_think_inputs(tok, row['prompt'], 'cuda')['input_ids']
                v = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -1].float().cpu()
                probability = torch.sigmoid(probe(v)).item()
                if not 0 <= probability <= 1:
                    raise ValueError('Non-finite probability')
                rows.append({k: v for k, v in row.items() if k != 'prompt'} | {
                    'probability': probability, 'predicted_answerable': probability >= 0.5})
                if (i + 1) % 48 == 0:
                    print(json.dumps({'encoded': i + 1, 'total': len(inputs)}), flush=True)
        after = {k: sha(v) for k, v in paths.items()}
        result = {'rows': rows, 'input_hashes': before, 'input_hashes_after': after,
                  'dataset_sha256': dataset_sha, 'seconds': time.perf_counter() - started,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'limitations': ['One mapping of names and one framing change; synthetic reused cases.',
                                  'No generation, independent evaluation, or persistence.']}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        if before != after:
            raise ValueError('Protected input changed')
        print(json.dumps({'complete': True, 'seconds': result['seconds']}), flush=True)
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__':
    main()
