"""Greedy evaluation on held-out synthetic object/place combinations."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer
from scripts.converse_research import prefix_tokens
from scripts.prepare_conversation_v2 import sha
from spora.conversation import generate, restore_conversation
from spora.growing import load
from spora.language_training import frozen_digest


def canonical(text):
    return re.sub(r"[\s。.!！?？]", "", text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((ROOT / args.plan).read_text(encoding="utf8"))
    dataset = ROOT / "runs/context-curriculum-v2"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf8"))
    source = dataset / "context-development.jsonl"
    if sha(source) != manifest["files"][source.name]:
        raise ValueError("Development set changed")
    vocab = ROOT / plan["data"] / "tokenizer.json"
    if sha(vocab) != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs")
    checkpoint = ROOT / args.checkpoint
    before_checkpoint = sha(checkpoint)
    parent = ROOT / plan["parent"]
    if sha(parent) != plan["parent_sha256"]:
        raise ValueError("Individual changed")
    torch.set_num_threads(2)
    organism, state, _ = load(parent, "cuda")
    before_individual = frozen_digest(organism, state)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = restore_conversation(organism, payload, plan["parent_sha256"], sha(vocab), "cuda")
    tokenizer = Tokenizer.from_file(str(vocab))
    eos = tokenizer.token_to_id("<|eos|>")
    forbidden = tuple(tokenizer.token_to_id(s) for s in ("<|pad|>", "<|bos|>", "<|user|>", "<|assistant|>"))
    rows = []
    for line in source.open(encoding="utf8"):
        example = json.loads(line)
        messages = example["messages"]
        prefix = prefix_tokens(tokenizer, [(m["role"], m["content"]) for m in messages[:-1]])
        result = generate(model, state, prefix, eos, budget=24, forbidden=forbidden)
        text = tokenizer.decode(result["tokens"], skip_special_tokens=False)
        expected = messages[-1]["content"]
        rows.append({"group": example["group"], "expected": expected, "output": text,
                     "ended": result["ended_with_eos"], "exact": canonical(text) == canonical(expected)})
    if before_checkpoint != sha(checkpoint) or before_individual != frozen_digest(organism, state):
        raise AssertionError("Read-only evaluation changed state")
    report = {"scope": "Held-out synthetic object/place combinations; supplied assistant history; not open chat acceptance.",
              "checkpoint_sha256": before_checkpoint, "development_sha256": sha(source),
              "evaluator_sha256": sha(Path(__file__)), "rows": rows,
              "exact": sum(r["exact"] for r in rows), "total": len(rows),
              "ended": sum(r["ended"] for r in rows)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({k: report[k] for k in ("exact", "total", "ended")}))


if __name__ == "__main__":
    main()
