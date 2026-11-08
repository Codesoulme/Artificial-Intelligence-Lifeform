"""Combine two previously reviewed sources without editing either source."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_conversation_v2 import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic", choices=("all-replies", "last-reply"), default="all-replies")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    authored = ROOT / "runs/p35-authored-dialogue-v4"
    synthetic = ROOT / ("runs/context-curriculum-v2" if args.synthetic == "all-replies"
                        else "runs/context-curriculum-last-v1")
    synthetic_hash = ("0620a6ff3da17c6381c24c7447a046bb4e8e62fd3bee537dd2a49e72bca634ae"
                      if args.synthetic == "all-replies"
                      else "b983e84f5b65922029583cd02543fadda4986b4a3957d5edc9fa192eb7078409")
    expected = {
        authored / "manifest.json": "00e023c9c08e674b289fb3419551e790ebc784f7a093b33fca3745c24f96956b",
        synthetic / "manifest.json": synthetic_hash,
    }
    for path, value in expected.items():
        if sha(path) != value:
            raise ValueError("Source manifest changed: " + str(path))
        manifest = json.loads(path.read_text(encoding="utf8"))
        for name, digest in manifest["files"].items():
            if sha(path.parent / name) != digest:
                raise ValueError("Source artifact changed: " + name)
    args.output.mkdir(parents=True)
    for source, target in ((authored / "teacher.jsonl", "teacher.jsonl"),
                           (authored / "teacher-tokens.jsonl", "teacher-tokens.jsonl"),
                           (synthetic / "teacher.jsonl", "foundations.jsonl"),
                           (synthetic / "teacher-tokens.jsonl", "foundations-tokens.jsonl")):
        shutil.copy2(source, args.output / target)
    vocab = ROOT / "runs/conversation-data-v5/tokenizer.json"
    manifest = {"status": "combined_candidate_pending_review", "tokenizer_sha256": sha(vocab),
                "source_manifests": {str(p.relative_to(ROOT)): h for p, h in expected.items()},
                "generator_sha256": sha(Path(__file__)),
                "files": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
                "limitations": ["Authored data is 24 conversations; synthetic data has only three templates.",
                                "The combination does not establish broad chat competence."]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
