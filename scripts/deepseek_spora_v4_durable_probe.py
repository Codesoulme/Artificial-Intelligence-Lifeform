"""Create an isolated incremental store from the real v4 bootstrap."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from spora.integrated_durable import create_integrated_durable
from spora.language_training import frozen_digest

BOOTSTRAP = ROOT/'runs/deepseek-spora-memory-output-integrated-v5-20261111/individual-with-output-adapter.pt'
OUTPUT = ROOT/'runs/deepseek-spora-v4-durable-probe-20261111.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Base mismatch')
    torch.set_num_threads(2)
    before = sha(BOOTSTRAP)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
                    trust_remote_code=False, torch_dtype=torch.float16,
                    attn_implementation='sdpa').to('cuda').eval()
    with tempfile.TemporaryDirectory(prefix='spora-v4-durable-') as temporary:
        database = Path(temporary)/'individual.sqlite'
        with create_integrated_durable(database, BOOTSTRAP, base, MODEL,
                                       individual_device='cuda') as runtime:
            previous_cursor = runtime.state.cursor
            decision, stats = runtime.observe_sourced(9, 3, 'direct', 1.0)
            organism, state, _, revision = runtime.store.read(device='cuda')
            identical = frozen_digest(organism, state) == frozen_digest(runtime.organism, runtime.state)
            result = {'initial_cursor': previous_cursor, 'final_cursor': state.cursor,
                      'revision': revision, 'decision': decision,
                      'incremental_tensor_bytes': stats['tensor_bytes_written'],
                      'state_digest_equal': identical,
                      'output_adapter_loaded': runtime.decoder.output_adapter is not None}
    result.update({'bootstrap_unchanged': sha(BOOTSTRAP) == before,
                   'bootstrap_sha256': before,
                   'limitations': ['Temporary isolated store; no user memory was changed.',
                                   'Same-process store read, not a real-model crash recovery test.',
                                   'Not integrated with the interactive chat entry.']})
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
