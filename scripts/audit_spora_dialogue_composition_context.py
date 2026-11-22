"""Diagnose the effects of the model's earlier replies and turn boundaries."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.audit_spora_chat_backbones import FILES, reply
from scripts.audit_spora_native_query_key import write
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_dialogue_composition import DATA, parse_answer, score_rollouts, settings_for
from spora.integrated_durable import load_integrated_durable
from spora.language_training import frozen_digest


def labeled(generated, case):
    score = parse_answer(generated['answer'], case)
    return {'generated': generated, 'score': score, 'passed': score['complete'] and generated['eos']}


def compare(model, tokenizer, cases, settings, rollout, minimum_tokens):
    rows = {}
    if minimum_tokens is not None:
        controlled = SimpleNamespace(generate=partial(model.generate, min_new_tokens=minimum_tokens))
    for case in cases:
        name = case['name']
        first, second, last = case['turns']
        if [row['question'] for row in rollout[name]['turns']] != case['turns']:
            raise ValueError('The recorded conversation does not match the sealed evaluation')
        actual = rollout[name]['turns']
        scores = {'actual': labeled(actual[-1], case)}
        if minimum_tokens is not None:
            history = [{'role': role, 'content': row[field]} for row in actual[:2]
                       for role, field in (('user', 'question'), ('assistant', 'answer'))]
            scores['minimum'] = labeled(reply(controlled, tokenizer, history, last, **settings), case)
        else:
            blank = [{'role': 'user', 'content': first}, {'role': 'assistant', 'content': ''},
                     {'role': 'user', 'content': second}, {'role': 'assistant', 'content': ''}]
            combined = '\n'.join((first, second, last))
            scores['blank'] = labeled(reply(model, tokenizer, blank, last, **settings), case)
            scores['combined'] = labeled(reply(model, tokenizer, [], combined, **settings), case)
        rows[name] = scores
    print({mode: sum(row[mode]['passed'] for row in rows.values()) for mode in next(iter(rows.values()))},
          flush=True)
    return rows


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    cases = json.loads(DATA.read_text(encoding='utf-8'))['evaluation']
    torch.set_num_threads(2)
    if args.qwen is not None:
        sources = [DATA, *(args.qwen / name for name in FILES)]
        protected = {path: sha(path) for path in sources}
        tokenizer = AutoTokenizer.from_pretrained(args.qwen, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(args.qwen, local_files_only=True, trust_remote_code=False,
                                                    torch_dtype=torch.float16, attn_implementation='sdpa')
        model.to('cuda').eval().requires_grad_(False)
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        settings = {'device': str(model.device), 'deepseek': False, 'limit': 64, 'eos': eos}
        rollout = score_rollouts(model, tokenizer, cases, settings=settings)
        rows = compare(model, tokenizer, cases, settings, rollout, args.minimum_answer_tokens)
    else:
        sources = [DATA, args.checkpoint, args.database, args.record]
        protected = {path: sha(path) for path in sources}
        recorded = json.loads(args.record.read_text(encoding='utf-8'))
        expected = recorded.get('checkpoint_sha256') if args.phase == 'after' else None
        if expected is None:
            expected = next((value for path, value in recorded['sources'].items()
                             if Path(path).resolve() == args.checkpoint.resolve()), None)
        if protected[args.checkpoint] != expected:
            raise ValueError('Use the checkpoint corresponding to the recorded rollout')
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        with load_integrated_durable(args.database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
            digest = frozen_digest(runtime.organism, runtime.state)
            if digest != recorded['individual_digest']:
                raise ValueError('The saved individual differs from the recorded checkpoint')
            settings = settings_for(runtime.decoder, tokenizer, 64)
            rows = compare(runtime.decoder, tokenizer, cases, settings, recorded[args.phase],
                           args.minimum_answer_tokens)
            if frozen_digest(runtime.organism, runtime.state) != digest:
                raise ValueError('Context audit changed the neural individual')
    if any(sha(path) != value for path, value in protected.items()):
        raise ValueError('Context audit changed a source file')
    write(args.output, {'rows': rows, 'minimum_answer_tokens': args.minimum_answer_tokens,
                        'sources': {str(path): value for path, value in protected.items()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'database', 'record', 'qwen', 'output'):
        parser.add_argument('--' + name, type=Path, required=name == 'output')
    parser.add_argument('--phase', choices=('before', 'after'), default='after')
    parser.add_argument('--minimum-answer-tokens', type=int, help='Test delayed EOS with the recorded histories.')
    args = parser.parse_args()
    if args.qwen is None and any(path is None for path in (args.checkpoint, args.database, args.record)):
        parser.error('Supply --qwen or the SPORA checkpoint, database and recorded run')
    if args.minimum_answer_tokens is not None and not 1 <= args.minimum_answer_tokens < 64:
        parser.error('--minimum-answer-tokens must be between 1 and 63')
    main(args)
