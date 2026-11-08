"""Compare fixed development generations from P139/P140 without external judging."""
import json,re
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
RUNS={"p139":ROOT/"runs/p139-authored-pretrain-smoke/development.json","p140":ROOT/"runs/p140-authored-pretrain-seed2-smoke/development.json"}
PROMPTS=6

def metrics(run):
 data=json.loads(run.read_text(encoding="utf-8")); last=data[-1]; rows=last["samples"]
 loops=0; lengths=[]; ended=0; utf8=0; repeated=[]
 for r in rows:
  text=r.get("text",""); lengths.append(len(text)); ended+=bool(r.get("ended_with_eos")); utf8+=bool(r.get("utf8_valid",False))
  lines=[x.strip() for x in text.splitlines() if x.strip()]
  direct=bool(re.search(r"(.{2,20})\1{3,}",text))
  numbered=bool(re.search(r"(?:^|\n)\s*\d+[.、)]",text))
  loops+=direct or numbered; repeated.append(round(len(text)/max(1,len(set(text))),3))
 return {"step":last["step"],"rows":len(rows),"eos":ended,"utf8":utf8,"obvious_loop":loops,"lengths":lengths,"unique_char_ratio_mean":round(sum(repeated)/len(repeated),4),"pretrain_nll":last["pretrain_nll"],"dialogue_nll":last["dialogue_nll"]}

def run():
 result={name:metrics(path) for name,path in RUNS.items()}
 out=ROOT/"runs/p141-authored-generation-audit-20261109";out.mkdir(exist_ok=False)
 (out/"summary.json").write_text(json.dumps({"status":"development_generation_audit_not_acceptance","runs":result,"limitations":["Fixed six prompts are development material, not independent acceptance.","Heuristics do not judge usefulness, truth, safety or multi-turn coherence.","No external model or final evaluation was used."]},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 return result

if __name__=="__main__":print(json.dumps(run(),ensure_ascii=False,indent=2))
