"""Run evidence-bounded pairs through the unchanged integrated SPORA copy."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.prepare_dulemon_evidence_pairs_20261117 import OUT as DATA
from scripts.deepseek_dulemon_evidence_context_baseline_20261117 import INSTRUCTION
from spora.language_training import frozen_digest

OUT = ROOT / 'runs/deepseek-dulemon-evidence-spora-long-context-20261117.json'


def check_position_budget(prompt_length, generation_budget, position_limits):
    if (type(prompt_length) is not int or prompt_length <= 0 or type(generation_budget) is not int
            or generation_budget <= 0 or not isinstance(position_limits, dict) or not position_limits
            or any(type(x) is not int or x <= 0 for x in position_limits.values())):
        raise ValueError('Invalid position budget')
    if prompt_length + generation_budget > min(position_limits.values()):
        raise ValueError('Prompt and generation budget exceed replacement position range')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {'base': runtime.MODEL/'model.safetensors', 'checkpoint': runtime.CHECKPOINT, 'data': DATA}
    hashes = {k: runtime.sha(v) for k, v in paths.items()}
    if hashes['base'] != runtime.WEIGHT_SHA:
        raise ValueError('Base mismatch')
    data = json.loads(DATA.read_text(encoding='utf-8'))
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(prefix='spora-long-context-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        position_limits = {i: chat.base.model.layers[i].max_length for i in (6, 12)}
        digest = frozen_digest(chat.organism, chat.state)
        rows = []
        with torch.inference_mode():
            for row in data['rows']:
                conversation = '\n'.join(f"{turn['speaker']}：{turn['text']}" for turn in row['context'])
                prompt = closed_think_inputs(chat.tokenizer,
                    INSTRUCTION + conversation + '\n问题：' + row['question'], 'cuda')
                check_position_budget(int(prompt['input_ids'].shape[1]), 32, position_limits)
                generated = chat.decoder.generate(**prompt, max_new_tokens=32, do_sample=False,
                    pad_token_id=chat.tokenizer.pad_token_id, eos_token_id=chat.tokenizer.eos_token_id)
                ids = generated[0, prompt['input_ids'].shape[1]:].tolist()
                rows.append({'session': row['session'], 'condition': row['condition'],
                             'output': chat.tokenizer.decode(ids, skip_special_tokens=True).strip(),
                             'token_ids': ids, 'prompt_tokens': int(prompt['input_ids'].shape[1]),
                             'terminated': bool(ids and ids[-1] == chat.tokenizer.eos_token_id)})
                print(json.dumps({'completed': len(rows), 'total': len(data['rows'])}), flush=True)
        if frozen_digest(chat.organism, chat.state) != digest:
            raise ValueError('Original state changed')
        if any(runtime.sha(v) != hashes[k] for k, v in paths.items()):
            raise ValueError('Protected source changed')
        result = {'status': 'integrated_spora_context_only_read_only', 'position_limits': position_limits,
                  'rows': rows, 'source_sha256': hashes,
                  'seconds': time.perf_counter()-started,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'limitations': ['No position extension or training performed.',
                                  'Addresses, facts and natural-language memory are not used in this run.',
                                  'Eight manually selected development pairs; not an independent benchmark.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'rows': len(rows), 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
