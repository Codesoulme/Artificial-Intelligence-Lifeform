"""Evaluate a saved dialogue model on composition and open development prompts."""

import argparse
import json
from pathlib import Path

from scripts.audit_local_named_chat import run
from scripts.audit_named_dialogue_midlayers_holdout import CASES
from scripts.audit_spora_chat_backbones import DEVELOPMENT, evaluate
from scripts.deepseek_local_baseline import MODEL, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_dialogue_composition import DATA, score_rollouts, settings_for
from spora.integrated_checkpoint import load_integrated
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--short-holdout', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {str(path): sha(path) for path in (args.checkpoint, DATA, DEVELOPMENT)}
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder, _, _ = load_integrated(args.checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    settings = settings_for(decoder, tokenizer, 96)
    composition = json.loads(DATA.read_text(encoding='utf-8'))['evaluation']
    development = json.loads(DEVELOPMENT.read_text(encoding='utf-8'))['scenarios']
    result = {'sources': sources}
    if args.short_holdout:
        result['short'] = run(decoder, tokenizer, cases=CASES)
        print('Short holdout', {key: value for key, value in result['short'].items() if key != 'rows'}, flush=True)
    result['composition'] = score_rollouts(decoder, tokenizer, composition, settings=settings)
    result['open'], _ = evaluate(decoder, tokenizer, development, {}, settings)
    if any(sha(Path(path)) != digest for path, digest in sources.items()):
        raise ValueError('Evaluation input changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
