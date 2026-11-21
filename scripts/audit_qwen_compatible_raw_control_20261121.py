"""Compare raw DeepSeek, all-compatible SPORA, and active replacements."""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_awake_feedforward_memory import HISTORY, history_cases
from audit_awake_multiturn_layers import CASES, ceval_questions, conversation, score_ceval, score_memory
from deepseek_output_aligned_memory import MODEL, sha, torch, write
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated


RAW_CHECKPOINT = Path("runs/qwen-compatible-raw-v1/integrated.pt")
ACTIVE_CHECKPOINT = Path("runs/qwen-compatible-persistent-v2/integrated.pt")
CEVAL_SUBJECTS = ("high_school_chinese", "logic", "middle_school_mathematics")


def evaluate_memory(decoder, tokenizer, organism, *, paired_history=False):
    _, cases = history_cases(organism)
    rows = score_memory(decoder, tokenizer, cases)
    result = {"memory": rows}
    summary = {"memory_correct": sum(row["correct"] for row in rows), "memory_total": len(rows)}
    if paired_history:
        updated_history = tuple((key, name, (earlier + 2) % 8, (current + 2) % 8)
                                for key, name, earlier, current in HISTORY)
        _, updated_cases = history_cases(organism, updated_history)
        alternate = score_memory(decoder, tokenizer, updated_cases)
        result["alternate_memory"] = alternate
        summary["memory_switches"] = sum(
            fixed["correct"] and changed["correct"] and fixed["answer"] != changed["answer"]
            and fixed["expected"] not in changed["answer"] and changed["expected"] not in fixed["answer"]
            for fixed, changed in zip(rows, alternate, strict=True)
        )
    return result, summary


def evaluate(decoder, tokenizer, ceval, position_limit, organism=None, *, paired_history=False):
    questions = list(ceval_questions(ceval, tokenizer))
    ceval_rows = score_ceval(decoder, questions, position_limit)
    dialogue_rows = [conversation(decoder, tokenizer, case, position_limit,
                                   decoding={"limit": 48}) for case in CASES]
    if organism is None and paired_history:
        raise ValueError("Paired history requires a memory organism")
    memory, memory_summary = ({"memory": []}, {"memory_correct": 0, "memory_total": 0})
    if organism is not None:
        memory, memory_summary = evaluate_memory(decoder, tokenizer, organism, paired_history=paired_history)
    result = {
        "ceval": ceval_rows,
        "dialogue": dialogue_rows,
        **memory,
        "summary": {
            "ceval_correct": sum(row["choice"] == row["expected"] for row in ceval_rows),
            "ceval_eligible": sum(row["choice"] is not None for row in ceval_rows),
            "dialogue_first_correct": sum(rows[0]["correct"] for rows in dialogue_rows),
            "dialogue_followup_correct": sum(rows[1]["correct"] for rows in dialogue_rows),
            "dialogue_both_correct": sum(all(row["correct"] for row in rows) for rows in dialogue_rows),
            "dialogue_eos": sum(row["eos"] for rows in dialogue_rows for row in rows),
            **memory_summary,
        },
    }
    return result


def main(output: Path, ceval: Path):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    protected = {
        RAW_CHECKPOINT: sha(RAW_CHECKPOINT),
        MODEL / "model.safetensors": sha(MODEL / "model.safetensors"),
        MODEL / "tokenizer.json": sha(MODEL / "tokenizer.json"),
    }
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    decoder, organism, _ = load_integrated(RAW_CHECKPOINT, base, MODEL, individual_device="cpu")
    decoder.eval().requires_grad_(False)
    position_limit = min(
        getattr(layer, "max_length", base.config.max_position_embeddings)
        for layer in base.model.layers
    )
    results = {"raw_deepseek": evaluate(base, tokenizer, ceval, base.config.max_position_embeddings),
               "all_compatible_spora": evaluate(decoder, tokenizer, ceval, position_limit, organism)}
    print("raw_deepseek", results["raw_deepseek"]["summary"], flush=True)
    print("all_compatible_spora", results["all_compatible_spora"]["summary"], flush=True)
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError("Protected source changed during audit")
    write(output, {
        "format": "spora-qwen-compatible-raw-control-v1",
        "raw_checkpoint": str(RAW_CHECKPOINT),
        "active_checkpoint": str(ACTIVE_CHECKPOINT),
        "base_model": str(MODEL),
        "ceval_subjects": CEVAL_SUBJECTS,
        "results": results,
        "seconds": time.perf_counter() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "sources": {str(path): digest for path, digest in protected.items()},
        "limitations": [
            "The active two-layer replacement checkpoint is referenced for provenance; this run evaluates the raw and all-compatible conditions.",
            "The all-compatible condition preserves native Qwen computation and is not a full SPORA block replacement.",
        ],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ceval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.ceval.is_dir():
        parser.error("C-Eval directory does not exist")
    main(args.output, args.ceval)
