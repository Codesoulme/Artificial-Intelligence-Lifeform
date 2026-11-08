"""Compare authored pretrain, full SFT and 5% mixed smoke outputs."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
RUNS={"P139_pretrain":"runs/p139-authored-pretrain-smoke/development.json","P142_full_sft":"runs/p142-authored-dialogue-sft-smoke/development.json","P143_mixed_5pct":"runs/p143-authored-mixed-5pct-smoke/development.json"}

def read(path):
 d=json.loads((ROOT/path).read_text(encoding="utf-8")); last=d[-1]
 ended=sum(bool(x.get("ended_with_eos")) for x in last["samples"]); loops=sum(bool(re.search(r"(.{2,20})\1{3,}",x.get("text",""))) for x in last["samples"])
 return {"step":last["step"],"pretrain_nll":last["pretrain_nll"],"dialogue_nll":last["dialogue_nll"],"eos":ended,"obvious_loops":loops,"utf8":sum(bool(x.get("utf8_valid")) for x in last["samples"]),"lengths":[len(x.get("text","")) for x in last["samples"]]}

def run():
 out=ROOT/"runs/p144-authored-route-comparison-20261109";out.mkdir(exist_ok=False)
 result={name:read(path) for name,path in RUNS.items()}
 payload={"status":"route_comparison_not_chat_acceptance","routes":result,"limitations":["Six fixed prompts are development material.","Heuristics do not judge usefulness, truth or multi-turn coherence.","No final evaluation or external judge was used."]}
 (out/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 return payload

if __name__=="__main__":print(json.dumps(run(),ensure_ascii=False,indent=2))
