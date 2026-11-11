"""Persist the trained output adapter with the candidate structural checkpoint."""
import json
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_composed_router_answer_eval import _load_candidate
from scripts.deepseek_spora_memory_answer_distill import SOURCE_CHECKPOINT
from scripts.deepseek_spora_recalibrate_name_router import SOURCE
from spora.integrated_checkpoint import (load_integrated, replacement_feature_variant_sha256,
                                          save_integrated)
from spora.language_training import atomic_save, frozen_digest
from spora.memory_output import MemoryOutputAdapter

ADAPTER_DIR = ROOT/'runs/deepseek-spora-memory-output-adapter-20261111'
OUTPUT = ROOT/'runs/deepseek-spora-memory-output-integrated-v5-20261111'
CHECKPOINT = OUTPUT/'individual-with-output-adapter.pt'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base mismatch')
    adapter_report = json.loads((ADAPTER_DIR/'result.json').read_text(encoding='utf-8'))
    adapter_path = ADAPTER_DIR/'adapter.pt'
    if sha(adapter_path) != adapter_report['adapter_sha256']:
        raise ValueError('Output adapter digest mismatch')
    adapter_payload = torch.load(adapter_path, map_location='cpu', weights_only=True)
    if (adapter_payload.get('format') != 'spora-memory-output-adapter-v1'
            or adapter_payload.get('base_sha256') != WEIGHT_SHA):
        raise ValueError('Output adapter provenance mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    decoder, organism, state = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cpu')
    blocks = {i: _load_candidate(SOURCE/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    variant = replacement_feature_variant_sha256(MODEL, blocks)
    adapter = MemoryOutputAdapter(**adapter_payload['config'])
    adapter.load_state_dict(adapter_payload['state'], strict=True)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    save_integrated(CHECKPOINT, organism, state, decoder.reader, MODEL, (5, 17),
                    replacements=blocks, output_adapter=adapter)
    adapter_config = adapter.config_dict()
    adapter_state = {name: tensor.detach().cpu().clone() for name, tensor in adapter.state_dict().items()}
    del decoder, base, blocks, adapter
    gc.collect()
    torch.cuda.empty_cache()
    restored_base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                         torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    restored, restored_organism, restored_state = load_integrated(CHECKPOINT, restored_base, MODEL,
                                                                   individual_device='cpu')
    if restored.output_adapter is None or restored.output_adapter.config_dict() != adapter_config:
        raise ValueError('Persisted output adapter configuration mismatch')
    for name, tensor in adapter_state.items():
        if not torch.equal(restored.output_adapter.state_dict()[name].cpu(), tensor):
            raise ValueError(f'Persisted output adapter tensor mismatch: {name}')
    if frozen_digest(organism, state) != frozen_digest(restored_organism, restored_state):
        raise ValueError('Persisted individual state mismatch')
    report = {'status': 'integrated_output_adapter_checkpoint', 'base_sha256': WEIGHT_SHA,
              'source_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
              'candidate_source': str(SOURCE), 'feature_variant_sha256': variant,
              'adapter_sha256': sha(adapter_path), 'checkpoint_sha256': sha(CHECKPOINT),
              'checkpoint': str(CHECKPOINT), 'adapter_config': adapter_config,
              'elapsed_seconds': time.perf_counter()-started,
              'limitations': ['Checkpoint persists the finite memory output adapter and candidate layers.',
                              'Runtime still needs a routed memory key to supply output evidence.',
                              'This artifact does not establish open-domain Chinese dialogue.',
                              'Base weights remain external and are checked by digest.']}
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('checkpoint_sha256', 'feature_variant_sha256', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    run()
