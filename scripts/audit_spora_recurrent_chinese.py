"""Compare the current recurrent chat weights with the original and parent decoders."""

import argparse
from pathlib import Path
from time import perf_counter

from scripts.audit_spora_fresh_dialogue import EARLY_BUDGET, FINAL_BUDGET, POSITIONS, digest, evaluate
from scripts.benchmark_spora_recurrent_dialogue import extend_positions, using
from scripts.audit_spora_chat_backbones import FILES
from scripts.audit_spora_dialogue_evidence import generate, store_json
from scripts.deepseek_local_baseline import MODEL, AutoTokenizer, torch
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import frozen_digest
from spora.research_recurrent_decode import load_recurrent_replacements


def evaluate_arm(label, model, tokenizer):
    def answer(history, question, budget):
        result = generate(model, tokenizer, history, question, POSITIONS, budget=budget)
        if result['retained'] != len(history) or result['budget'] != budget:
            raise ValueError('An arm lost conversation history or generation budget')
        return result

    print(label, flush=True)
    return evaluate(answer)


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    manifest = Path(__file__).with_name('spora_fresh_dialogue_cases.py')
    sources = {path: digest(path) for path in (
        manifest, Path(__file__), args.checkpoint, *(MODEL / name for name in FILES),
        *(directory / name for directory in args.candidates for name in ('attention.pt', 'comparison.json')),
    )}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    with torch.inference_mode():
        native = evaluate_arm('native', base, tokenizer)
    base.lm_head.to('cpu')
    native_rotary = base.model.rotary_emb
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL, disposable_base=True)
    decoder = runtime.decoder.eval().requires_grad_(False)
    if decoder.position_table_max_position < 2048:
        extend_positions(decoder, native_rotary, 2048)
    individual = frozen_digest(runtime.organism, runtime.state)
    replacements = load_recurrent_replacements(decoder, args.candidates, sources[args.checkpoint])
    if tuple(replacements) != (19, 22):
        raise ValueError('Expected the two replacements used by the local chat entry point')
    torch.cuda.empty_cache()
    base.lm_head.to(base.device)
    with torch.inference_mode():
        parent = evaluate_arm('parent', base, tokenizer)
        with using(replacements, 'recurrent'):
            recurrent = evaluate_arm('recurrent', base, tokenizer)
    if (frozen_digest(runtime.organism, runtime.state) != individual
            or any(digest(path) != value for path, value in sources.items())):
        raise ValueError('A source file or the reconstructed individual changed during the audit')
    store_json(args.output, {
        'native': native, 'parent': parent, 'recurrent': recurrent,
        'layers': list(replacements), 'sources': {str(path): value for path, value in sources.items()},
        'cases_sha256': sources[manifest], 'individual_sha256': individual,
        'early_budget': EARLY_BUDGET, 'final_budget': FINAL_BUDGET, 'positions': POSITIONS,
        'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated(),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs=2, type=Path)
    main(parser.parse_args())
