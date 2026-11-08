"""Evaluate whether counterfactual Chinese context changes candidate ranking.

Development-only diagnostic: two hand-authored contexts per item and two
candidate completions. Does not establish open-ended chat competence.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from scripts.converse_research import prefix_tokens
from scripts.prepare_conversation_v2 import sha
from spora.conversation import restore_conversation
from spora.growing import load
from spora.language_training import frozen_digest

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("number", "篮子里现在有两本书。", "篮子里现在有五本书。", "篮子里有几本书？", "两本", "五本"),
    ("place", "小林今天去了公园。", "小林今天去了车站。", "小林今天去了哪里？", "公园", "车站"),
    ("color", "小红的杯子是蓝色的。", "小红的杯子是黄色的。", "小红的杯子是什么颜色？", "蓝色", "黄色"),
    ("time", "会议改到周一上午。", "会议改到周二上午。", "会议最终在星期几？", "周一", "周二"),
    ("animal", "圆圆是一只小猫。", "圆圆是一只小狗。", "圆圆是什么动物？", "小猫", "小狗"),
    ("object", "桌上放着一支笔。", "桌上放着一把伞。", "桌上放了什么？", "一支笔", "一把伞"),
    ("correction", "原来约在周一，后来改为周二。", "原来约在周二，后来改为周一。", "最后约在星期几？", "周二", "周一"),
    ("preference", "小明喜欢喝茶，不喜欢咖啡。", "小明喜欢咖啡，不喜欢喝茶。", "小明喜欢什么？", "茶", "咖啡"),
]


@torch.no_grad()
def score(model, state, tokenizer, prefix, answer):
    ids = tokenizer.encode(answer).ids + [tokenizer.token_to_id("<|eos|>")]
    tokens = torch.tensor([prefix + ids], dtype=torch.long, device=next(model.parameters()).device)
    logits = model(tokens[:, :-1], state)
    selected = logits[0, len(prefix) - 1 : len(prefix) - 1 + len(ids)]
    targets = torch.tensor(ids, device=tokens.device)
    return float(F.log_softmax(selected, -1).gather(-1, targets[:, None]).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads((ROOT / args.plan).read_text(encoding="utf8"))
    checkpoint = ROOT / args.checkpoint
    before_checkpoint = sha(checkpoint)
    parent = ROOT / plan["parent"]
    if sha(parent) != plan["parent_sha256"]:
        raise ValueError("Individual changed")
    vocab = ROOT / plan["data"] / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(vocab))
    torch.set_num_threads(2)
    organism, state, _ = load(parent, "cuda")
    before_individual = frozen_digest(organism, state)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = restore_conversation(organism, payload, plan["parent_sha256"], sha(vocab), "cuda")
    rows = []
    for name, left, right, question, answer_left, answer_right in CASES:
        pair = []
        for context, expected in ((left, answer_left), (right, answer_right)):
            prefix = prefix_tokens(tokenizer, [("user", context + question)])
            scores = [score(model, state, tokenizer, prefix, answer) for answer in (answer_left, answer_right)]
            winner = (answer_left, answer_right)[int(scores[1] > scores[0])]
            pair.append({"context": context, "question": question, "expected": expected,
                         "candidates": [answer_left, answer_right], "logprob": scores,
                         "winner": winner, "correct": winner == expected, "tie": scores[0] == scores[1]})
        rows.append({"name": name, "variants": pair,
                     "both_correct": all(x["correct"] and not x["tie"] for x in pair),
                     "winner_changes": pair[0]["winner"] != pair[1]["winner"]})
    if sha(checkpoint) != before_checkpoint or frozen_digest(organism, state) != before_individual:
        raise AssertionError("Read-only evaluation changed state")
    result = {"scope": "Hand-authored development-only two-choice likelihood diagnostic; not open-ended acceptance.",
              "checkpoint_sha256": before_checkpoint, "tokenizer_sha256": sha(vocab),
              "evaluator_sha256": sha(Path(__file__)), "rows": rows,
              "both_correct_pairs": sum(x["both_correct"] for x in rows),
              "winner_changed_pairs": sum(x["winner_changes"] for x in rows)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"both_correct_pairs": result["both_correct_pairs"],
                      "winner_changed_pairs": result["winner_changed_pairs"], "pairs": len(rows)}))


if __name__ == "__main__":
    main()
