"""Evaluate a saved output-aware writer against the entire language vocabulary."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_text_writer_broad_generalization as prior
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_output_aware_writer import GAIN, OUT as SOURCE, norm_vectors
from spora.semantic_text_writer import TextSemanticWriter, value_positions

OUT = ROOT / 'runs/deepseek-spora-output-aware-reload-20261116.json'


def main(source=SOURCE, output=OUT):
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    report = json.loads((source / 'result.json').read_text(encoding='utf-8'))
    path = source / 'writer.pt'
    digest = base.sha(path)
    if report['writer_sha256'] != digest or report['base_sha256'] != base.WEIGHT_SHA:
        raise ValueError('Writer provenance mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    writer = TextSemanticWriter(model.config.hidden_size).to('cuda').eval()
    writer.load_state_dict(payload['state'], strict=True)
    rows = []
    for name_split, names in (('train', prior.TRAIN_NAMES), ('held', prior.HELD_NAMES)):
        for name in names:
            prompt = pilot.closed_think_inputs(tokenizer, f'{name}喜欢喝什么饮料？', 'cuda')
            with torch.inference_mode():
                prompt_hidden = model.model(**prompt, use_cache=False, return_dict=True).last_hidden_state[:, -1]
            for value_split in ('train', 'held'):
                values = report[f'{value_split}_values'][:8]
                for value in values:
                    encoded, positions = value_positions(tokenizer, f'{name}喜欢喝{value}。', value)
                    if len(positions) != 1:
                        raise ValueError('Expected one value token')
                    encoded = {k: v.to('cuda') for k, v in encoded.items()}
                    token = int(encoded['input_ids'][0, positions[0]])
                    with torch.inference_mode():
                        hidden = model.model(**encoded, use_cache=False, output_hidden_states=True,
                                             return_dict=True).hidden_states[1][0, positions[0]].float()
                        vector = norm_vectors(writer(hidden), payload['train_norm'])
                        logits = model.lm_head(prompt_hidden + (GAIN * vector).to(prompt_hidden.dtype)).float()[0]
                        rank = int((logits > logits[token]).sum()) + 1
                    rows.append({'name': name, 'value': value, 'name_split': name_split,
                                 'value_split': value_split, 'target': token, 'rank': rank,
                                 'top1': int(logits.argmax())})
            print(json.dumps({'completed_name_split': name_split, 'cases': len(rows)}), flush=True)
    if base.sha(path) != digest or base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA:
        raise ValueError('Source changed')
    summary = {}
    for n in ('train', 'held'):
        for v in ('train', 'held'):
            subset = [r for r in rows if r['name_split'] == n and r['value_split'] == v]
            summary[f'{n}_name_{v}_value'] = {'count': len(subset),
                                               'top1': sum(r['rank'] == 1 for r in subset)}
    result = {'summary': summary, 'rows': rows, 'base_sha256': base.WEIGHT_SHA, 'writer_sha256': digest,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['First eight values per split, two names per split, fixed artificial question.',
                              'Teacher-forced first token only; no answer or EOS generation here.',
                              'Program supplies value boundaries; no autonomous memory routing.',
                              'No SPORA replacement layer or persistent individual.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    main(args.source, args.output)
