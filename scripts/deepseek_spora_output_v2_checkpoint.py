"""Build and independently verify an isolated output-v2 structural checkpoint."""
import gc
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))
import torch
from transformers import AutoModelForCausalLM
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_runtime_chat import CHECKPOINT
from spora.integrated_checkpoint import load_integrated, save_integrated, replacement_feature_variant_sha256
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputAdapter
from spora.structure_transfer import SPORALayerReplacement

SOURCE = ROOT/'runs/deepseek-spora-memory-output-adapter-v2-20261111'
OUT = ROOT/'runs/deepseek-spora-output-v2-checkpoint-20261111'
TARGET = OUT/'individual-with-output-v2.pt'


def base():
    return AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
               trust_remote_code=False, torch_dtype=torch.float16,
               attn_implementation='sdpa').to('cuda').eval()


def inspect(path):
    model = base()
    decoder, organism, state = load_integrated(path, model, MODEL, individual_device='cpu')
    if decoder.output_adapter is None:
        raise ValueError('Output adapter missing')
    return {'individual_digest': frozen_digest(organism, state), 'cursor': state.cursor,
            'adapter_config': decoder.output_adapter.config_dict(),
            'adapter_tensors': {k: v.detach().cpu() for k, v in decoder.output_adapter.state_dict().items()}}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Base mismatch')
    report = json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))
    if sha(SOURCE/'adapter.pt') != report['adapter_sha256']:
        raise ValueError('Adapter digest mismatch')
    payload = torch.load(SOURCE/'adapter.pt', map_location='cpu', weights_only=True)
    if payload.get('format') != 'spora-memory-output-adapter-v1' or payload.get('base_sha256') != WEIGHT_SHA:
        raise ValueError('Adapter provenance mismatch')
    torch.set_num_threads(2)
    source_hash = sha(CHECKPOINT)
    model = base()
    decoder, organism, state = load_integrated(CHECKPOINT, model, MODEL, individual_device='cpu')
    replacements = {i: layer for i, layer in enumerate(model.model.layers)
                    if isinstance(layer, SPORALayerReplacement)}
    router_report = json.loads((ROOT/'runs/deepseek-spora-recalibrate-name-router-20261111/result.json').read_text(encoding='utf-8'))
    if replacement_feature_variant_sha256(MODEL, replacements) != router_report['feature_variant_sha256']:
        raise ValueError('Loaded structural variant mismatch')
    # Enforce the same source layers on which this adapter was trained.
    for index in (6, 12):
        path = ROOT/'runs/deepseek-spora-diverse-memory-replay-20261111'/f'layer-{index}.pt'
        if sha(path) != payload['candidate_sha256'][str(index)]:
            raise ValueError('Adapter structural source mismatch')
    adapter = MemoryOutputAdapter(**payload['config'])
    adapter.load_state_dict(payload['state'], strict=True)
    before_digest = frozen_digest(organism, state)
    OUT.mkdir(parents=False, exist_ok=False)
    save_integrated(TARGET, organism, state, decoder.reader, MODEL, decoder.layer_indices,
                    replacements=replacements, output_adapter=adapter)
    expected = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    del model, decoder, organism, state, replacements, adapter
    gc.collect()
    torch.cuda.empty_cache()
    # Use a new interpreter to prove the persisted artifact can be restored.
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--inspect', str(TARGET)],
                           capture_output=True, timeout=240, check=True)
    info = json.loads(child.stdout.decode('utf-8').strip().splitlines()[-1])
    if info['individual_digest'] != before_digest or not info['adapter_equal']:
        raise ValueError('Independent restore mismatch')
    if sha(CHECKPOINT) != source_hash:
        raise ValueError('Source checkpoint changed')
    result = {'status': 'isolated_candidate_checkpoint', 'source_sha256': source_hash,
              'candidate_checkpoint_sha256': sha(TARGET), 'candidate_adapter_sha256': report['adapter_sha256'],
              'individual_digest': before_digest, 'independent_restore': info,
              'limitations': ['Checkpoint still requires the external DeepSeek base.',
                              'Separate candidate; interactive runtime was not switched.',
                              'Parameter/state restore is not an end-to-end dialogue evaluation.']}
    with (OUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('source_sha256','candidate_checkpoint_sha256','independent_restore')}), flush=True)


def child(path):
    torch.set_num_threads(2)
    info = inspect(path)
    payload = torch.load(SOURCE/'adapter.pt', map_location='cpu', weights_only=True)
    info['adapter_equal'] = all(torch.equal(info['adapter_tensors'][k], v.cpu())
                                for k, v in payload['state'].items())
    del info['adapter_tensors']
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--inspect':
        child(sys.argv[2])
    elif len(sys.argv) == 1:
        run()
    else:
        raise SystemExit('Invalid arguments')
