"""Replay failed dialogue revisions on intact and integrated instruction weights."""

import argparse
from collections import Counter
from functools import partial
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_spora_dialogue_evidence import BUDGET, CASES, POSITIONS, scores, store_json
from scripts.audit_named_memory_qwen_structure_control import (
    CHECKPOINT_SHA, QWEN_SHA, attach_control, load_calibration,
)
from scripts.audit_spora_chat_backbones import FILES, reply
from scripts.deepseek_local_baseline import AutoTokenizer, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_qwen_streamed
from scripts.spora_dialogue_revision_data import HELD, ORACLE_PROMPTS
from spora.integrated_checkpoint import _digest
from spora.language_training import frozen_digest


def evaluate(generate, cases):
    rows = []
    counts = Counter()
    for name, stages, clean_prompt in cases:
        history, rollout = [], []
        for stage in stages:
            text = stage['text']
            result = generate(history, text)
            rollout.append(result)
            history.extend([{'role': 'user', 'content': text},
                            {'role': 'assistant', 'content': result['answer']}])
        conditions = {'raw': rollout[-1], 'resolved': generate([], clean_prompt)}
        for label, result in conditions.items():
            result['score'] = scores(result['answer'], stages[-1])
            counts[f'{label}_pass'] += result['score']['pass']
            counts[f'{label}_eos'] += result['eos']
        rows.append({'name': name, 'rollout': rollout, 'conditions': conditions})
        print(name, {label: result['score']['pass'] for label, result in conditions.items()}, flush=True)
    return {'counts': dict(counts), 'rows': rows}


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    files = (args.checkpoint, *(args.donor / name for name in FILES))
    sources = {str(path): sha(path) for path in files}
    if sources[str(args.checkpoint)] != CHECKPOINT_SHA or sources[str(args.donor / 'model.safetensors')] != QWEN_SHA:
        raise ValueError('Expected the pinned instruction-backbone control')
    calibration_file = args.calibration.with_suffix('.reply.pt')
    calibration_sha = sha(args.calibration)
    tokenizer = AutoTokenizer.from_pretrained(args.donor, local_files_only=True, trust_remote_code=False)
    base = load_qwen_streamed(args.donor, QWEN_SHA)
    eos = base.generation_config.eos_token_id

    def comparisons(model):
        print(type(model).__name__, flush=True)
        native_reply = partial(reply, model, tokenizer, device='cuda', deepseek=False, limit=BUDGET, eos=eos)

        def generate(history, text):
            output = native_reply(history, text)
            if output['prompt_tokens'] + BUDGET > POSITIONS:
                raise ValueError('Dialogue exceeded the original study position budget')
            return output

        holdout = [(name, stages, ORACLE_PROMPTS[name]) for name, _, stages, _ in HELD]
        return {'development': evaluate(generate, CASES), 'held': evaluate(generate, holdout)}

    native = comparisons(base)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    metadata, tensors = payload['metadata'], payload['tensors']
    if _digest(metadata, tensors) != payload['integrity_sha256']:
        raise ValueError('SPORA checkpoint integrity mismatch')
    decoder, organism, state = attach_control(base, metadata, tensors)
    before = frozen_digest(organism, state)
    reference = json.loads(args.calibration.read_text(encoding='utf-8'))
    decoder.output_adapter, adapter_sha = load_calibration(calibration_file, reference, sources, decoder, organism)
    integrated = comparisons(decoder)
    if (frozen_digest(organism, state) != before or sha(calibration_file) != adapter_sha
            or sha(args.calibration) != calibration_sha
            or any(sha(Path(path)) != digest for path, digest in sources.items())):
        raise ValueError('Evaluation changed a protected source')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    store_json(args.output, {'sources': sources, 'calibration_sha256': calibration_sha,
                             'adapter_sha256': adapter_sha, 'individual_sha256': before,
                             'budget': BUDGET, 'positions': POSITIONS, 'native': native, 'integrated': integrated,
                             'peak_bytes': torch.cuda.max_memory_allocated(), 'seconds': perf_counter() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('checkpoint', 'donor', 'calibration', 'output'):
        parser.add_argument(argument, type=Path)
    main(parser.parse_args())
