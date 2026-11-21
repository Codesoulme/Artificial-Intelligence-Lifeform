"""Measure a GELU gate transplant on the memory-enabled DeepSeek candidate."""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from audit_qwen_compatible_raw_control_20261121 import evaluate
from audit_awake_multiturn_layers import ceval_questions, score_ceval
from deepseek_output_aligned_memory import MODEL, sha, torch, write
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated_runtime
from spora.integrated_open_base import SPORAGatedFeedForward
from spora.pretrained_attention import SPORAGeluGate, SPORAResidualBlock
from transformers import AutoTokenizer


class ScaledGelu(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, gate):
        return (torch.nn.functional.gelu(gate.float() * self.scale) / self.scale).to(gate.dtype)


def fit_gate_scale(decoder, tokenizer, mlp):
    from train_awake_partial_layer import corpus

    analytic = math.sqrt(2. * math.pi) / 4.
    scales = (analytic, .55, .575, .60, .61, .62, .64, .65, .675, .70)
    texts = corpus()[:32]
    if len(texts) < 32:
        raise ValueError('Gate calibration requires 24 fit and 8 held-out texts')
    errors = torch.zeros((2, len(scales) + 1), device=mlp.down.weight.device)
    split = 0

    def capture(_module, _inputs, projected):
        gate, value = projected.chunk(2, dim=-1)
        target = mlp.down(mlp.activation(gate) * value).float()
        errors[split, -1] += target.square().sum()
        for index, scale in enumerate(scales):
            predicted = mlp.down(ScaledGelu(scale)(gate) * value)
            errors[split, index] += (predicted.float() - target).square().sum()

    handle = mlp.projection.register_forward_hook(capture)
    try:
        for index, text in enumerate(texts):
            split = int(index >= 24)
            tokens = tokenizer.encode(text, add_special_tokens=False)[:128]
            tokens = torch.tensor([tokens], device=mlp.down.weight.device)
            decoder(input_ids=tokens, use_cache=False, logits_to_keep=1)
    finally:
        handle.remove()
    if not torch.isfinite(errors).all() or (errors[:, -1] <= 0).any():
        raise ValueError('Invalid gate calibration error')
    errors = (errors[:, :-1] / errors[:, -1:]).cpu().tolist()
    best = min(range(len(scales)), key=errors[0].__getitem__)
    return {'fit_texts': 24, 'held_texts': 8, 'selected_scale': scales[best],
            'scores': [{'scale': scale, 'fit_relative_mse': fit, 'held_relative_mse': held}
                       for scale, fit, held in zip(scales, *errors, strict=True)]}


def fit_logit_scale(decoder, tokenizer, mlp):
    from train_awake_partial_layer import corpus

    analytic = math.sqrt(2. * math.pi) / 4.
    scales = (.48, .50, .52, .54, .55, .56, .58, .60, analytic)
    texts = corpus()[:16]
    if len(texts) < 16:
        raise ValueError('Logit calibration requires 12 fit and 4 held-out texts')
    original = mlp.activation
    errors = [[0.] * len(scales) for _ in range(2)]

    def predict(tokens):
        logits = decoder(input_ids=tokens, use_cache=False, logits_to_keep=4).logits
        return logits[0].float().log_softmax(dim=-1)

    try:
        for index, text in enumerate(texts):
            split = int(index >= 12)
            tokens = tokenizer.encode(text, add_special_tokens=False)[:96]
            tokens = torch.tensor([tokens], device=mlp.down.weight.device)
            mlp.activation = original
            reference = predict(tokens)
            for column, scale in enumerate(scales):
                mlp.activation = ScaledGelu(scale)
                estimate = predict(tokens)
                divergence = torch.nn.functional.kl_div(estimate, reference, reduction='batchmean', log_target=True)
                errors[split][column] += float(divergence)
    finally:
        mlp.activation = original
    if any(not math.isfinite(error) for row in errors for error in row):
        raise ValueError('Nonfinite logit calibration error')
    selected = min(range(len(scales)), key=errors[0].__getitem__)
    return {'fit_texts': 12, 'held_texts': 4, 'selected_scale': scales[selected],
            'scores': [{'scale': scale, 'fit_kl': fit / 12, 'held_kl': held / 4}
                       for scale, fit, held in zip(scales, *errors, strict=True)]}


class GeluGate(torch.nn.Module):
    """Use a scaled GELU that initially approximates the pretrained SiLU."""

    def __init__(self, original, fraction):
        super().__init__()
        self.original = original
        self.gelu = SPORAGeluGate()
        self.fraction = fraction

    def forward(self, gate):
        return self.original(gate).lerp(self.gelu(gate), self.fraction)


@torch.inference_mode()
def main(checkpoint, ceval, output, index, candidate=None, subjects=None, fit_scale=None):
    if output.exists():
        raise FileExistsError(output)
    source_digest = sha(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if candidate is not None and candidate.exists():
        raise FileExistsError(candidate)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL)
    decoder, organism = runtime.decoder, runtime.organism
    decoder.eval().requires_grad_(False)
    layer = decoder.base.model.layers[index]
    if index in decoder.layer_indices:
        layer = layer.layer
    if isinstance(layer, SPORAResidualBlock):
        layer = layer.base
    mlp = layer.mlp
    if not isinstance(mlp, SPORAGatedFeedForward):
        raise ValueError('Expected a compatible SPORA feed-forward network')
    original = mlp.activation
    if fit_scale and isinstance(original, SPORAGeluGate):
        original = torch.nn.SiLU().eval()
        mlp.activation = original
    if candidate is not None:
        mlp.activation = SPORAGeluGate().eval()
        candidate.parent.mkdir(parents=True, exist_ok=True)
        runtime.save(candidate)
        if sha(checkpoint) != source_digest:
            raise ValueError('Source checkpoint changed during migration')
        output.parent.mkdir(parents=True, exist_ok=True)
        write(output, {'layer': index, 'checkpoint_sha256': source_digest, 'candidate_sha256': sha(candidate)})
        return
    limit = decoder.position_table_max_position or decoder.base.config.max_position_embeddings
    if subjects is not None:
        questions = tuple(ceval_questions(ceval, tokenizer, subjects))
    fitters = {'local': fit_gate_scale, 'logits': fit_logit_scale}
    calibration = fitters[fit_scale](decoder, tokenizer, mlp) if fit_scale else None
    if calibration is not None:
        print('Gate calibration', calibration, flush=True)
        conditions = {'silu': original, 'analytic': SPORAGeluGate().eval(),
                      'calibrated': ScaledGelu(calibration['selected_scale']).eval()}
    elif subjects is not None:
        conditions = {'0.0': original, '1.0': SPORAGeluGate().eval()}
    else:
        conditions = {'0.0': original, '0.5': GeluGate(original, .5), '1.0': SPORAGeluGate().eval()}
    results = {}
    try:
        for name, activation in conditions.items():
            mlp.activation = activation
            if subjects is None:
                result = evaluate(decoder, tokenizer, ceval, limit, organism, paired_history=True)
            else:
                rows = score_ceval(decoder, questions, limit)
                result = {'ceval': rows, 'summary': {
                    'ceval_correct': sum(row['choice'] == row['expected'] for row in rows),
                    'ceval_total': len(rows),
                }}
            results[name] = result
            print(name, result['summary'], flush=True)
    finally:
        mlp.activation = original
    if sha(checkpoint) != source_digest:
        raise ValueError('Source checkpoint changed during evaluation')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'format': 'spora-single-layer-gelu-gate-audit-v1', 'layer': index,
              'checkpoint_sha256': source_digest, 'subjects': subjects, 'results': results}
    if calibration is not None:
        report['gate_calibration'] = calibration
        report['calibration_objective'] = fit_scale
    write(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--ceval', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=21)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--candidate', type=Path)
    mode.add_argument('--fit-scale', nargs='?', const='local', choices=('local', 'logits'))
    parser.add_argument('--subjects', nargs='+')
    args = parser.parse_args()
    main(args.checkpoint, args.ceval, args.output, args.layer, args.candidate, args.subjects, args.fit_scale)
