"""Make a copy of the synthetic curriculum supervised only on final replies."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenizers import Tokenizer
from scripts.prepare_conversation_v2 import sha

BASE = ROOT / "runs/context-curriculum-v2"
EXPECTED = "0620a6ff3da17c6381c24c7447a046bb4e8e62fd3bee537dd2a49e72bca634ae"


def final_reply_mask(tokens, mask, assistant_id):
    if len(tokens) != len(mask):
        raise ValueError("Token and mask lengths differ")
    locations = [i for i, token in enumerate(tokens) if token == assistant_id]
    if not locations or not any(mask[locations[-1] + 1:]):
        raise ValueError("Missing final supervised assistant reply")
    return [bool(value and i > locations[-1]) for i, value in enumerate(mask)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(BASE / "manifest.json") != EXPECTED:
        raise ValueError("Input curriculum manifest changed")
    manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf8"))
    for name, digest in manifest["files"].items():
        if sha(BASE / name) != digest:
            raise ValueError("Input artifact changed: " + name)
    vocab = ROOT / "runs/conversation-data-v5/tokenizer.json"
    if sha(vocab) != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer changed")
    assistant_id = Tokenizer.from_file(str(vocab)).token_to_id("<|assistant|>")
    args.output.mkdir(parents=True)
    for name in ("teacher.jsonl", "context-development.jsonl", "context-development-tokens.jsonl",
                 "foundations.jsonl", "foundations-tokens.jsonl"):
        shutil.copy2(BASE / name, args.output / name)
    targets = 0
    with (BASE / "teacher-tokens.jsonl").open(encoding="utf8") as src, \
         (args.output / "teacher-tokens.jsonl").open("x", encoding="utf8") as dst:
        for line in src:
            row = json.loads(line)
            row["mask"] = final_reply_mask(row["tokens"], row["mask"], assistant_id)
            targets += sum(row["mask"])
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    output = {"status": "candidate_pending_review", "source_manifest_sha256": EXPECTED,
              "tokenizer_sha256": sha(vocab), "generator_sha256": sha(Path(__file__)),
              "supervised_targets": targets,
              "files": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
              "limitations": ["Same 145 synthetic dialogues and three templates as P60.",
                              "Only final assistant reply is supervised; earlier turns remain visible context.",
                              "No claim of broad conversation competence."]}
    (args.output / "manifest.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
