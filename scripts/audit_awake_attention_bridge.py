"""Transplant pretrained attention into a single SPORA-style QKV projection."""

import argparse
import time
from pathlib import Path

from audit_awake_multiturn_layers import CASES, ceval_questions, encode, load_layer_pair, score_ceval, score_memory
from audit_awake_feedforward_memory import history_cases
from deepseek_output_aligned_memory import MODEL, SOURCE, read, sha, torch, write
from spora.integrated_open_base import SPORAGatedFeedForward
from train_awake_replacement_replay import generate
from spora.pretrained_attention import SPORAFusedAttention, load_absolute_attention


@torch.inference_mode()
def main(output, index, keep_groups=False, fuse_feedforward=False, ceval=None, absolute=None, position_table=False):
    before_digests = {path: sha(path) for path in (SOURCE, MODEL / 'model.safetensors')}
    start = time.perf_counter()
    decoder, tokenizer, _, replacements, organism, _ = load_layer_pair()
    if index is not None:
        if not 0 <= index < len(decoder.base.model.layers) or index in {*replacements, *decoder.layer_indices}:
            raise ValueError('Select an original decoder layer without a memory wrapper')
        layer = decoder.base.model.layers[index]
        origin = layer.self_attn
        mlp = layer.mlp
        if absolute is None:
            bridge = SPORAFusedAttention(origin, keep_groups=keep_groups).eval()
        else:
            before_digests[absolute] = read(absolute.parent / 'result.json')['candidate_sha256']
            bridge = load_absolute_attention(absolute, origin, before_digests[SOURCE], before_digests[absolute])
        fused_mlp = SPORAGatedFeedForward(mlp).eval() if fuse_feedforward else mlp
    _, memory = history_cases(organism)
    prompts = [encode(tokenizer, [dict(role='user', content=case[0])]) for case in CASES]
    position_limit = min(layer.max_length for layer in replacements.values())
    if ceval is not None:
        from audit_awake_multiturn_layers import CEVAL_SUBJECTS

        for subject in CEVAL_SUBJECTS:
            source = ceval / f'{subject}.parquet'
            before_digests[source] = sha(source)
        questions = list(ceval_questions(ceval, tokenizer))
        if absolute is not None:
            position_limit = min(position_limit, bridge.max_position)
    results = {}
    ceval_results = {}
    migrated = ()
    position_parameters = 0
    old_count = sum(parameter.numel() for parameter in decoder.parameters())
    for name in ('original', 'fused'):
        if name == 'fused':
            if index is None:
                migrated = decoder.migrate_pretrained_blocks()
            else:
                layer.self_attn = bridge
                layer.mlp = fused_mlp
            if position_table:
                table = decoder.migrate_position_encoder(position_limit)
                position_parameters = table.coefficients.weight.numel()
        rows = []
        for case, prompt in zip(CASES, prompts, strict=True):
            inputs = torch.tensor([prompt], device='cuda')
            logits = decoder(input_ids=inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
            tokens = generate(decoder, tokenizer, prompt, 32)
            rows.append((logits, tokens, tokenizer.decode(tokens, skip_special_tokens=True).strip()))
        memory_rows = score_memory(decoder, tokenizer, memory)
        results[name] = (rows, memory_rows)
        print(name, 'memory', sum(row['correct'] for row in memory_rows), '/', len(memory_rows), flush=True)
        if ceval is not None:
            ceval_results[name] = score_ceval(decoder, questions, position_limit)
            rows_ceval = ceval_results[name]
            print(name, 'C-Eval', sum(row['choice'] == row['expected'] for row in rows_ceval), '/',
                  sum(row['choice'] is not None for row in rows_ceval), flush=True)
    ordinary = [dict(question=case[0], logits_max_abs_difference=(old[0] - new[0]).abs().max().item(),
                     generation_identical=old[1] == new[1], original=old[2], fused=new[2])
                for case, old, new in zip(CASES, results['original'][0], results['fused'][0], strict=True)]
    if any(sha(path) != digest for path, digest in before_digests.items()):
        raise ValueError('A source checkpoint changed during the audit')
    new_count = sum(parameter.numel() for parameter in decoder.parameters())
    if index is not None:
        old_count = sum(parameter.numel() for parameter in origin.parameters())
        new_count = sum(parameter.numel() for parameter in bridge.parameters())
    write(output, dict(layer=index, migrated=migrated, keep_groups=keep_groups, fuse_feedforward=fuse_feedforward,
                       position_parameters=position_parameters,
                       ordinary=ordinary, memory={name: rows[1] for name, rows in results.items()},
                       ceval=ceval_results, original_parameters=old_count, fused_parameters=new_count,
                       sources={str(path): digest for path, digest in before_digests.items()},
                       seconds=time.perf_counter()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    print('Identical generations', sum(row['generation_identical'] for row in ordinary), '/', len(ordinary),
          'max logit difference', max(row['logits_max_abs_difference'] for row in ordinary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layer', type=int, default=18)
    parser.add_argument('--all', action='store_true', help='Convert all remaining pretrained layers')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--keep-groups', action='store_true', help='Preserve grouped key and value dimensions')
    parser.add_argument('--fuse-feedforward', action='store_true', help='Also fuse the pretrained feed-forward input projections')
    parser.add_argument('--ceval', type=Path, help='Include local C-Eval validation shards')
    parser.add_argument('--absolute', type=Path, help='Test a saved absolute-position attention candidate')
    parser.add_argument('--position-table', action='store_true', help='Also convert the shared position encoder')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose an unused output path')
    if args.absolute is not None and (args.all or args.keep_groups):
        parser.error('--absolute evaluates one attention module with ungrouped heads')
    main(args.output, None if args.all else args.layer, args.keep_groups or args.all,
         args.fuse_feedforward or args.all, args.ceval, args.absolute, args.position_table)
