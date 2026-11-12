"""Cross-process reload of experimental text-coded fast matrices, not runtime persistence."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_spora_continuous_text_capacity import sampled_sequences
from spora.continuous_text_memory import ContinuousTextCodec

OUTPUT = p.ROOT / 'runs/deepseek-spora-continuous-codec-restart-probe-20261112.json'
FORMAT = 'experimental-continuous-matrix-probe-v1'


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def worker(phase, directory):
    path = directory / 'matrices.pt'
    output = directory / (phase + '.json')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    p.torch.set_num_threads(2)
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    examples, _ = sampled_sequences(tokenizer)
    expected = [examples[16]] + examples[1:16]
    p.runtime.WORKING_CHECKPOINT = directory / 'unused-working.pt'
    chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
    if any(x.is_meta for x in list(chat.base.parameters()) + list(chat.base.buffers())):
        raise ValueError('Unmaterialized layer')
    organism = chat.organism.eval()
    original = p.frozen_digest(organism, chat.state)
    device = next(organism.parameters()).device
    codec = ContinuousTextCodec(chat.base.model.embed_tokens.weight.detach(),
        memory_width=organism.config.memory_width, slot_count=4, seed=29071, eos_token_id=tokenizer.eos_token_id)
    codebook_sha = tensor_sha(codec.codebook)
    projection_sha = tensor_sha(codec.projection)
    width = codec.slot_count * codec.slot_width
    if phase == 'write':
        fast, velocity = organism.initial_state(1)
        for index in range(len(fast)):
            fast[index][0] = torch.zeros_like(fast[index][0])
        mode = torch.zeros(1, dtype=torch.long, device=device)
        gate = torch.ones(1, device=device)
        for key, item in list(enumerate(examples[:16])) + [(0, examples[16])]:
            code = codec.encode(torch.tensor(item['token_ids'], dtype=torch.long, device=device))
            target = torch.zeros((1, organism.config.memory_width), device=device)
            target[0, :width] = code
            for index, site in enumerate(organism.memories):
                fast[index], velocity[index] = site.write_temporal(fast[index], velocity[index],
                    site.current_basis[key:key + 1].to(device), target, gate, mode, False)
        matrices = [site[0].detach().cpu().clone() for site in fast]
        if path.exists():
            raise FileExistsError(path)
        with path.open('xb') as stream:
            torch.save({'format': FORMAT, 'matrices': matrices, 'codebook_sha256': codebook_sha,
                        'projection_sha256': projection_sha, 'checkpoint_sha256': checkpoint_sha,
                        'base_sha256': p.WEIGHT_SHA, 'config': codec.config_dict()}, stream)
    elif phase == 'read':
        stored = torch.load(path, map_location='cpu', weights_only=True)
        if (stored['format'] != FORMAT or stored['checkpoint_sha256'] != checkpoint_sha
                or stored['base_sha256'] != p.WEIGHT_SHA or stored['codebook_sha256'] != codebook_sha
                or stored['projection_sha256'] != projection_sha or stored['config'] != codec.config_dict()):
            raise ValueError('Reloaded codec or source provenance mismatch')
        matrices = stored['matrices']
        if len(matrices) != len(organism.memories):
            raise ValueError('Unexpected memory site count')
    else:
        raise ValueError('Unknown phase')
    rows = []
    for key, item in enumerate(expected):
        observed = []
        for index, site in enumerate(organism.memories):
            matrix = matrices[index].to(device)
            with torch.no_grad():
                vector = site.retrieve([matrix], site.current_basis[key:key + 1].to(device))[0, :width]
            decoded = codec.decode(vector)
            observed.append(decoded['token_ids'])
        target = item['token_ids'] + [tokenizer.eos_token_id]
        rows.append({'key': key, 'expected_token_ids': target, 'site_token_ids': observed,
                     'both_exact': all(tokens == target for tokens in observed)})
    if p.frozen_digest(organism, chat.state) != original:
        raise ValueError('Original individual changed')
    if p.sha(p.runtime.CHECKPOINT) != checkpoint_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected input changed')
    report = {'phase': phase, 'rows': rows, 'exact_both': sum(r['both_exact'] for r in rows),
              'matrix_sha256': [tensor_sha(matrix) for matrix in matrices], 'codebook_sha256': codebook_sha,
              'projection_sha256': projection_sha, 'checkpoint_sha256': checkpoint_sha,
              'file_sha256': p.sha(path), 'file_bytes': path.stat().st_size}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'phase': phase, 'exact_both': report['exact_both']}), flush=True)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with tempfile.TemporaryDirectory() as directory:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        for phase in ('write', 'read'):
            command = [sys.executable, '-m', 'scripts.deepseek_spora_continuous_codec_restart_probe',
                       '--phase', phase, '--directory', directory]
            subprocess.run(command, cwd=str(p.ROOT), env=env, check=True)
        first = json.loads((Path(directory) / 'write.json').read_text(encoding='utf-8'))
        second = json.loads((Path(directory) / 'read.json').read_text(encoding='utf-8'))
        if first['rows'] != second['rows'] or first['matrix_sha256'] != second['matrix_sha256'] or first['file_sha256'] != second['file_sha256']:
            raise ValueError('Cross-process reload mismatch')
        result = {'before': first, 'after': second, 'cross_process_match': True,
                  'limitations': ['Custom temporary experimental file, not the SPORA runtime checkpoint format.',
                                  'Sixteen generated token sequences, manual numeric addresses and one correction.',
                                  'Fixed codebook regenerated from the same base and seed; no portability guarantee.',
                                  'No natural-language routing, provenance, history or crash-interruption test.',
                                  'No actual individual or checkpoint modified.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'cross_process_match': True, 'exact_both': first['exact_both']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('write', 'read'))
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    if args.phase:
        if args.directory is None:
            parser.error('--directory required for worker')
        worker(args.phase, args.directory)
    else:
        main()
