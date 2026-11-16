"""Isolated, multi-process persistence and correction audit for a learned writer.

State is a separate vector-slot file; identity, fact boundaries and read timing
are supplied by a small rule-based research interface, not learned by the model.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from scripts.deepseek_spora_output_aware_writer import GAIN, norm_vectors
from spora.research_semantic_slot import install_sequence_scaled, use_slot_sequence
from spora.semantic_phrase_slot_runtime import NaturalSemanticPhraseMemory
from spora.semantic_phrase_slot_state import SemanticPhraseSlotState
from spora.semantic_slot_runtime import parse_fact
from spora.semantic_text_writer import TextSemanticWriter, value_positions

SOURCE = ROOT / 'runs/deepseek-spora-output-aware-broad-20261116'
OUT = ROOT / 'runs/deepseek-spora-broad-writer-persistence-20261116'
INITIAL = ('小赵喜欢喝咖啡。', '小郑喜欢喝红茶。')
CORRECTION = '小赵喜欢喝椰汁。'
UNKNOWN = '小吴喜欢喝什么饮料？'


def binding(base_sha, writer_sha, layer_hashes):
    data = {'base': base_sha, 'writer': writer_sha, 'layers': layer_hashes, 'gain': GAIN}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()


def worker(phase, directory):
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    paths = {i: PAIR / f'layer-{i}.pt' for i in (6, 12)}
    if (base.sha(SOURCE / 'writer.pt') != manifest['writer_sha256']
            or any(base.sha(path) != manifest['layer_sha256'][str(i)] for i, path in paths.items())):
        raise ValueError('Writer or replacement input changed')
    payload = torch.load(SOURCE / 'writer.pt', map_location='cpu', weights_only=True)
    if payload['base_sha256'] != base.WEIGHT_SHA:
        raise ValueError('Writer base mismatch')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed(skip_replaced=(6, 12)).eval()
    for index, path in paths.items():
        model.model.layers[index] = _load_candidate(path, index, model).eval()
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError('Unmaterialized parameter')
    writer = TextSemanticWriter(model.config.hidden_size).to('cuda').eval()
    writer.load_state_dict(payload['state'], strict=True)
    install_sequence_scaled(model, scale=GAIN)
    state_path = directory / 'isolated-state.pt'
    bridge = binding(base.WEIGHT_SHA, manifest['writer_sha256'], manifest['layer_sha256'])
    if phase == 'initial':
        if state_path.exists():
            raise FileExistsError(state_path)
        memory = NaturalSemanticPhraseMemory(SemanticPhraseSlotState(model.config.hidden_size, 4, max_tokens=8),
                                             base_sha256=base.WEIGHT_SHA, bridge_sha256=bridge)
    else:
        memory = NaturalSemanticPhraseMemory.load(state_path, expected_base_sha256=base.WEIGHT_SHA,
                                                   expected_bridge_sha256=bridge)

    def write_fact(text):
        fact = parse_fact(text)
        encoded, positions = value_positions(tokenizer, text, fact.value)
        encoded = {key: value.to('cuda') for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model.model(**encoded, use_cache=False, output_hidden_states=True,
                                 return_dict=True).hidden_states[1][0, positions].float()
            vectors = norm_vectors(writer(hidden), payload['train_norm']).detach().cpu()
        return memory.write_encoded_sequence(text, vectors, token_count=len(positions))

    def evaluate():
        rows = []
        for name in ('小赵', '小郑'):
            question = f'{name}喜欢喝什么饮料？'
            route = memory.query_slot(question)
            if route['slot'] is None:
                raise ValueError('Known slot missing')
            prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
            slots = route['value'].to('cuda')
            eos = model.model.embed_tokens.weight[tokenizer.eos_token_id].float().unsqueeze(0)
            modes = {}
            for mode in ('oracle_eos', 'natural_continuation'):
                supplied = torch.cat((slots, eos), dim=0) if mode == 'oracle_eos' else slots
                with torch.inference_mode(), use_slot_sequence(supplied):
                    generated = model.generate(**prompt, use_cache=True, do_sample=False, max_new_tokens=16,
                                               pad_token_id=tokenizer.pad_token_id,
                                               eos_token_id=tokenizer.eos_token_id)
                tail = generated[0, prompt['input_ids'].shape[1]:].tolist()
                modes[mode] = {'ids': tail, 'text': tokenizer.decode(tail, skip_special_tokens=True).strip(),
                               'ended': bool(tail and tail[-1] == tokenizer.eos_token_id)}
            digest = hashlib.sha256(route['value'].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            rows.append({'name': name, 'slot': route['slot'], 'slot_sha256': digest, 'modes': modes})
        unknown = memory.query_slot(UNKNOWN)
        return {'rows': rows, 'unknown_no_slot': unknown['slot'] is None}

    writes = []
    before = None
    if phase == 'initial':
        writes = [write_fact(text) for text in INITIAL]
        memory.save(state_path, metadata={'audit': 'broad-writer-persistence-v1'})
    elif phase == 'correction':
        before = evaluate()
        writes = [write_fact(CORRECTION)]
        memory.save(state_path, metadata={'audit': 'broad-writer-persistence-v1'})
    elif phase != 'verify':
        raise ValueError('Invalid phase')
    result = {'phase': phase, 'before': before, 'after': evaluate(), 'writes': writes,
              'revision': memory.state.revision, 'state_sha256': base.sha(state_path),
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA
            or base.sha(SOURCE / 'writer.pt') != manifest['writer_sha256']
            or any(base.sha(path) != manifest['layer_sha256'][str(i)] for i, path in paths.items())):
        raise ValueError('Protected source changed')
    with (directory / f'{phase}.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'phase': phase, 'revision': result['revision']}), flush=True)


def main(output=OUT):
    if output.exists():
        raise FileExistsError(output)
    source_report = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    writer_sha = base.sha(SOURCE / 'writer.pt')
    if source_report['writer_sha256'] != writer_sha or source_report['base_sha256'] != base.WEIGHT_SHA:
        raise ValueError('Writer source mismatch')
    layer_hashes = {str(i): base.sha(PAIR / f'layer-{i}.pt') for i in (6, 12)}
    output.mkdir()
    manifest = {'writer_sha256': writer_sha, 'base_sha256': base.WEIGHT_SHA, 'layer_sha256': layer_hashes}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    for phase in ('initial', 'correction', 'verify'):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', phase,
                        '--directory', str(output.resolve())], cwd=ROOT, check=True)
    records = {phase: json.loads((output / f'{phase}.json').read_text(encoding='utf-8'))
               for phase in ('initial', 'correction', 'verify')}
    first_slots = {row['name']: row['slot_sha256'] for row in records['initial']['after']['rows']}
    corrected_slots = {row['name']: row['slot_sha256'] for row in records['correction']['after']['rows']}
    checks = {'initial_reload_answers_match': records['initial']['after'] == records['correction']['before'],
              'corrected_reload_answers_match': records['correction']['after'] == records['verify']['after'],
              'corrected_reload_file_match': records['correction']['state_sha256'] == records['verify']['state_sha256'],
              'changed_fact_slot_changed': first_slots['小赵'] != corrected_slots['小赵'],
              'other_fact_slot_unchanged': first_slots['小郑'] == corrected_slots['小郑'],
              'revisions': [records[x]['revision'] for x in ('initial', 'correction', 'verify')]}
    result = {'checks': checks, 'manifest': manifest,
              'limitations': ['Oracle value spans, names and read timing; hand-authored fixed phrases.',
                              'Vector-slot file is separate from base model weights and is not autonomous learning.',
                              'Three short phases; no open-domain chat, interruption or concurrent writes.',
                              'Oracle EOS conditions explicitly provide a correct stop vector.',
                              'Known test facts were previously examined during development.']}
    with (output / 'summary.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('initial', 'correction', 'verify'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    if args.worker:
        if args.directory is None:
            raise ValueError('Worker directory required')
        worker(args.worker, args.directory)
    else:
        main(args.output)
