"""Deterministic synthetic development curriculum for context-conditioning.

No external model outputs or real personal information. The generated held-out
combination split is for development diagnostics, not chat acceptance.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenizers import Tokenizer
from scripts.prepare_conversation_v2 import encode_dialogue, sha

OBJECTS = ("雨伞", "钥匙", "手套", "围巾", "杯子", "笔记本", "画笔", "水壶")
PLACES = ("木箱", "书架", "抽屉", "背包", "柜子", "篮子", "盒子", "桌边")


def rows():
    for oi, pi in itertools.product(range(len(OBJECTS)), range(len(PLACES))):
        obj, place = OBJECTS[oi], PLACES[pi]
        # Disjoint object/place combinations, rather than random turns.
        split = "development" if (oi * 11 + pi * 7) % 5 == 0 else "train"
        statements = (
            (f"我把{obj}放在{place}。", f"{obj}现在在哪里？"),
            (f"请记一下，{obj}的位置是{place}。", f"{obj}放在哪儿？"),
        )
        for variant, (statement, question) in enumerate(statements):
            yield split, f"recall-{oi}-{pi}-{variant}", [
                {"role": "user", "content": statement},
                {"role": "assistant", "content": "好的。"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": place + "。"},
            ]
        # A cyclic destination prevents position of the correct place being fixed.
        new_place = PLACES[(pi + 3) % len(PLACES)]
        new_pi = (pi + 3) % len(PLACES)
        correction_split = ("development" if split == "development" or
                            (oi * 11 + new_pi * 7) % 5 == 0 else "train")
        yield correction_split, f"correction-{oi}-{pi}", [
            {"role": "user", "content": f"我先把{obj}放在{place}。"},
            {"role": "assistant", "content": "记下了。"},
            {"role": "user", "content": f"改一下，我现在把{obj}放在{new_place}。"},
            {"role": "assistant", "content": "好的，位置改了。"},
            {"role": "user", "content": f"{obj}最终放在哪里？"},
            {"role": "assistant", "content": new_place + "。"},
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = ROOT / "runs/teaching-curriculum-v5"
    vocab = ROOT / "runs/conversation-data-v5/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(vocab))
    samples = {"train": [], "development": []}
    for split, group, messages in rows():
        encoded = encode_dialogue(tokenizer, messages, limit=512)
        if encoded is None:
            raise ValueError("Generated dialogue too long")
        ids, mask = encoded
        samples[split].append(({"group": group, "messages": messages},
                               {"group": group, "tokens": ids, "mask": mask}))
    args.output.mkdir(parents=True)
    for split, values in samples.items():
        prefix = "teacher" if split == "train" else "context-development"
        for suffix, column in ((".jsonl", 0), ("-tokens.jsonl", 1)):
            with (args.output / (prefix + suffix)).open("x", encoding="utf8") as stream:
                for row in values:
                    stream.write(json.dumps(row[column], ensure_ascii=False) + "\n")
    for name in ("foundations.jsonl", "foundations-tokens.jsonl"):
        shutil.copy2(base / name, args.output / name)
    manifest = {"status": "synthetic_candidate_pending_review", "source": "deterministic authored templates",
                "tokenizer_sha256": sha(vocab), "generator_sha256": sha(Path(__file__)),
                "counts": {key: len(value) for key, value in samples.items()},
                "files": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
                "limitations": ["Generated templates are narrow synthetic exercises, not real conversation.",
                                "Development holds out object/place combinations, not all vocabulary or templates.",
                                "No claim of broad semantic generalization or chat readiness."]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
