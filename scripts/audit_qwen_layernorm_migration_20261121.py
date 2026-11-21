"""Fit and evaluate a centered SPORA normalization in one compatible DeepSeek block."""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from audit_awake_block_components import fit_norm
from audit_awake_multiturn_layers import ceval_questions, score_ceval
from audit_qwen_compatible_raw_control_20261121 import evaluate
from audit_qwen_residual_memory_holdout_20261121 import memory_cases, score_state_pair
from deepseek_output_aligned_memory import MODEL, sha, torch, write
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import atomic_save
from spora.pretrained_attention import SPORAResidualBlock
from train_awake_partial_layer import TRAIN_FILES, collect, corpus, relative_mse
from train_qwen_compatible_residual_block_20261121 import expanded_memory_cases
from transformers import AutoTokenizer


class TrainableLayerNorm(torch.nn.LayerNorm):
    def forward(self, inputs):
        return super().forward(inputs.float()).to(inputs.dtype)


def refine_norm(decoder, block, original, candidate, examples, steps, memory_target_weight):
    fitted = TrainableLayerNorm(candidate.normalized_shape, eps=candidate.eps,
                                device=candidate.weight.device, dtype=torch.float32)
    fitted.load_state_dict(candidate.state_dict())
    optimizer = torch.optim.AdamW(fitted.parameters(), lr=2e-5, weight_decay=0.)
    random.Random(20261121).shuffle(examples)
    losses = []
    try:
        with torch.inference_mode(False), torch.enable_grad():
            for step in range(steps):
                prefix, target, controls = examples[step % len(examples)]
                device = fitted.weight.device
                tokens = torch.tensor([prefix + target[:-1]], device=device)
                count = len(target)
                block.post_attention_layernorm = original
                with torch.no_grad():
                    reference = decoder(input_ids=tokens, logits_to_keep=count,
                                        **dict(controls, use_cache=False)).logits[0].float().log_softmax(-1)
                block.post_attention_layernorm = fitted
                prediction = decoder(input_ids=tokens, logits_to_keep=count,
                                     **dict(controls, use_cache=False)).logits[0].float()
                loss = torch.nn.functional.kl_div(prediction.log_softmax(-1), reference,
                                                  reduction='batchmean', log_target=True)
                if controls and memory_target_weight:
                    labels = torch.tensor(target, device=device)
                    memory_loss = torch.nn.functional.cross_entropy(prediction, labels)
                    loss = loss + memory_target_weight * memory_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fitted.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                if (step + 1) % 12 == 0:
                    print('LayerNorm alignment', step + 1, 'mean loss', sum(losses[-12:]) / 12, flush=True)
    finally:
        block.post_attention_layernorm = original
    return fitted.requires_grad_(False).eval(), losses


def collect_memory_norm(decoder, tokenizer, organism, norm):
    features, targets = [], []

    def capture(_module, args, result):
        features.append(args[0][0].float().cpu())
        targets.append(result[0].float().cpu())

    handle = norm.register_forward_hook(capture)
    try:
        rows = expanded_memory_cases(organism, tokenizer)
        for prefix, target, controls in rows[::8] + rows[1::8]:
            tokens = torch.tensor([prefix + target[:-1]], device=norm.weight.device)
            decoder(input_ids=tokens, logits_to_keep=1, **dict(controls, use_cache=False))
    finally:
        handle.remove()
    return torch.cat(features).cuda(), torch.cat(targets).cuda()


@torch.no_grad()
def main(checkpoint, output, index, ceval, additional_ceval, *, memory_replay=False, refine_steps=0,
         memory_target_weight=0.):
    if output.exists():
        raise FileExistsError(output)
    protected = {path: sha(path) for path in (checkpoint, *TRAIN_FILES)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL)
    decoder, organism = runtime.decoder, runtime.organism
    decoder.eval().requires_grad_(False)
    if index not in decoder.compatible_block_indices:
        raise ValueError('Expected a compatible SPORA layer')
    block = decoder.base.model.layers[index]
    if index in decoder.layer_indices:
        block = block.layer
    if isinstance(block, SPORAResidualBlock):
        block = block.base
    original = block.post_attention_layernorm
    texts = corpus()
    if len(texts) < 40:
        raise ValueError('Expected at least 40 distinct calibration texts')
    features, targets = collect(decoder, tokenizer, texts[:32], original)
    memory_positions = 0
    if memory_replay:
        memory_features, memory_targets = collect_memory_norm(decoder, tokenizer, organism, original)
        memory_positions = len(memory_features)
        features = torch.cat((features, memory_features))
        targets = torch.cat((targets, memory_targets))
        del memory_features, memory_targets
    candidate = fit_norm(original, features, targets)
    del features, targets
    features, targets = collect(decoder, tokenizer, texts[32:40], original)
    error = relative_mse(lambda x: candidate(x.to(original.weight.dtype)).float(), features, targets)
    del features, targets
    print('Held-out normalization relative MSE', error, flush=True)
    losses = []
    if refine_steps:
        examples = expanded_memory_cases(organism, tokenizer)
        for text in texts[:32]:
            ids = tokenizer.encode(text, add_special_tokens=False)[:112]
            if len(ids) > 12:
                examples.append((ids[:-8], ids[-8:], {}))
        candidate, losses = refine_norm(decoder, block, original, candidate, examples, refine_steps,
                                        memory_target_weight)
    position_limit = decoder.position_table_max_position or decoder.base.config.max_position_embeddings
    subjects = ('high_school_geography', 'middle_school_history', 'middle_school_physics', 'modern_chinese_history')
    questions = tuple(ceval_questions(additional_ceval, tokenizer, subjects))
    current, changed = (memory_cases(organism, offset) for offset in (0, 2))
    results = {}
    try:
        for name, module in (('rmsnorm', original), ('layernorm', candidate)):
            block.post_attention_layernorm = module
            result = evaluate(decoder, tokenizer, ceval, position_limit, organism, paired_history=True)
            rows = score_ceval(decoder, questions, position_limit)
            result['additional_ceval'] = rows
            result['summary']['additional_ceval_correct'] = sum(row['choice'] == row['expected'] for row in rows)
            result['summary']['additional_ceval_total'] = len(rows)
            result['memory_holdout'] = score_state_pair(decoder, tokenizer, current, changed)
            results[name] = result
            print(name, result['summary'], result['memory_holdout']['summary'], flush=True)
    finally:
        block.post_attention_layernorm = original
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('An experiment source changed during the audit')
    output.parent.mkdir(parents=True, exist_ok=True)
    weights_sha = None
    if refine_steps:
        weights = output.with_suffix('.pt')
        if weights.exists():
            raise FileExistsError(weights)
        atomic_save({'layer': index, 'component': 'pre_feedforward', 'source_sha256': protected[checkpoint],
                     'epsilon': candidate.eps, 'weights': candidate.cpu().state_dict()}, weights)
        weights_sha = sha(weights)
    write(output, {'layer': index, 'component': 'pre_feedforward', 'train_texts': 32, 'held_texts': 8,
                   'memory_fit_positions': memory_positions,
                   'refine_steps': refine_steps, 'refine_loss': losses, 'candidate_sha256': weights_sha,
                   'memory_target_weight': memory_target_weight,
                   'held_relative_mse': error, 'results': results,
                   'sources': {str(path): digest for path, digest in protected.items()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=17)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--additional-ceval', type=Path, required=True)
    parser.add_argument('--memory-replay', action='store_true')
    parser.add_argument('--refine-steps', type=int, default=0)
    parser.add_argument('--memory-target-weight', type=float, default=0.)
    args = parser.parse_args()
    if args.refine_steps < 0:
        parser.error('--refine-steps must be nonnegative')
    if not 0 <= args.memory_target_weight <= 1 or (args.memory_target_weight and not args.refine_steps):
        parser.error('--memory-target-weight requires refinement and must be in [0, 1]')
    main(args.checkpoint, args.output, args.layer, args.ceval, args.additional_ceval,
         memory_replay=args.memory_replay, refine_steps=args.refine_steps,
         memory_target_weight=args.memory_target_weight)
