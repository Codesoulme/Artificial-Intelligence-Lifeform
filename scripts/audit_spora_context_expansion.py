"""Verify existing positions and exercise the extended SPORA checkpoint on CUDA."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.audit_spora_local_attention import answer, cache_probe, memory_answers
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_qwen_counterfactual_reply_20261122 import HELD, prepare, records
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import frozen_digest
from spora.research_local_attention import anchored_cache
from spora.value_position_checkpoint import VALUE_POSITION_PREFIX


def verify_tables(parent_path, new_path):
    parent, new = (torch.load(path, map_location='cpu', weights_only=True) for path in (parent_path, new_path))
    prior, current = parent['metadata']['position_table_max_position'], new['metadata']['position_table_max_position']
    before, after = parent['tensors'], new['tensors']
    if (current <= prior or before.keys() != after.keys()
            or new['metadata'] != dict(parent['metadata'], position_table_max_position=current)):
        raise ValueError('The expanded checkpoint changed more than the position limit')
    position_keys = {f'{VALUE_POSITION_PREFIX}{index}/position' for index in parent['metadata']['value_position_layers']}
    for key, tensor in before.items():
        if key in position_keys:
            if (after[key].shape[0] != current or not torch.equal(tensor, after[key][:prior])
                    or torch.count_nonzero(after[key][prior:])):
                raise ValueError('The learned value table changed existing positions')
        elif not torch.equal(tensor, after[key]):
            raise ValueError('A checkpoint tensor changed outside the position tables')
    return prior, current


@torch.inference_mode()
def old_positions(runtime, prose):
    decoder = runtime.decoder
    rows = {}
    for length in (128, 512, 768):
        ids = torch.tensor([prose[:length]], device='cuda')
        rows[length] = decoder(input_ids=ids, use_cache=False, logits_to_keep=1).logits.cpu()
    return rows


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    prior, current = verify_tables(args.parent, args.checkpoint)
    prose_path = ROOT / 'data/chinese-pilot/source.json'
    foundations = ROOT / 'data/chinese-foundations.json'
    sources = {str(path): sha(path) for path in (args.parent, args.checkpoint, prose_path, foundations)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    data = json.loads(prose_path.read_text(encoding='utf-8'))
    scenes = json.loads(foundations.read_text(encoding='utf-8'))['dialogues']
    extra = [content for scene in scenes for pair in scene['turns'] for content in pair]
    prose = tokenizer.encode('\n'.join(data['train'] + data['development'] + data['test'] + extra),
                             add_special_tokens=False)
    if len(prose) < current or prior != 768 or current != 1536:
        raise ValueError('Expected source prose and checkpoints for the 768-to-1536 audit')
    parent = load_integrated_runtime(args.parent, load_full_streamed(), MODEL)
    parent.decoder.eval()
    individual = frozen_digest(parent.organism, parent.state)
    baseline = old_positions(parent, prose)
    del parent
    gc.collect()
    torch.cuda.empty_cache()
    expanded = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL)
    decoder = expanded.decoder.eval()
    comparison = old_positions(expanded, prose)
    prefix = {length: float((baseline[length] - prediction).abs().max())
              for length, prediction in comparison.items()}
    if any(prefix.values()) or individual != frozen_digest(expanded.organism, expanded.state):
        raise ValueError('Existing positions or the original individual changed')
    layout = decoder.anchored_attention_layout

    def make_cache():
        return anchored_cache(decoder.base.config, layout['layers'], layout['window'], layout['prefix'])

    cache_rows = []
    for length in (768, 769, 896, 1024, current):
        row, _ = cache_probe(decoder, torch.tensor([prose[:length]], device='cuda'), make_cache)
        cache_rows.append(row)
    recall = []
    for offset, location in ((0, '栖水站'), (360, '柏岩站'), (950, '云河站')):
        text = (f'{tokenizer.decode(prose[:offset])}\n\n接驳通知的集合点是{location}。\n\n'
                f'{tokenizer.decode(prose[offset:1000])}\n\n请只回答接驳通知里的集合点。')
        row = answer(decoder, tokenizer, text)
        recall.append(dict(row, expected=location, offset=offset, correct=location in row['text']))
    fixtures = records(expanded.organism, HELD)
    controls = [prepare(expanded.organism, fixtures[i]) for i in (0, 1, 6, 7, 16, 17, 22, 23)]
    memory = memory_answers(decoder, tokenizer, tokenizer.decode(prose[:1000]), controls)
    if individual != frozen_digest(expanded.organism, expanded.state) or any(
        sha(Path(path)) != digest for path, digest in sources.items()
    ):
        raise ValueError('The extension audit modified a frozen source')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, {'prefix_max_logit_delta': prefix, 'position_limit': current, 'cache': cache_rows,
                        'recall': recall, 'memory': memory, 'individual_sha256': individual, 'sources': sources,
                        'seconds': time.perf_counter() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()})
    print('Verified positions, continuation, and long memory', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
