"""Add a user-owned historical SPORA document to a new candidate version."""
import hashlib,json,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/p136-authored-chinese-candidate-20261109/candidate.jsonl"
HISTORY=Path(r"E:\人工智能生命体\历史版本\SPORA-v1.0.docx")
OUT=ROOT/"runs/p145-authored-chinese-history-candidate-20261109"
NS={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def paragraphs(p):
 with zipfile.ZipFile(p) as z:r=ET.fromstring(z.read("word/document.xml"))
 return ["".join(t.text or "" for t in p.findall(".//w:t",NS)).strip() for p in r.findall(".//w:p",NS) if "".join(t.text or "" for t in p.findall(".//w:t",NS)).strip()]

def run():
 if OUT.exists():raise FileExistsError(OUT)
 if not HISTORY.exists():raise FileNotFoundError(HISTORY)
 base=[json.loads(x) for x in BASE.read_text(encoding="utf-8").splitlines() if x.strip()]
 hsha=sha(HISTORY); added=[]
 for i,text in enumerate(paragraphs(HISTORY)):
  if len(text)>=20:added.append({"record_id":f"user_architecture_history_docx:{i}","kind":"user_architecture_history_docx","source_file":str(HISTORY),"source_file_sha256":hsha,"text":text,"review_status":"user_authored_not_quality_reviewed"})
 OUT.mkdir(parents=False);q=OUT/"candidate.jsonl"
 with q.open("x",encoding="utf-8",newline="\n") as f:
  for row in base+added:f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
 result={"status":"candidate_not_trained_authored_history_added","base_records":len(base),"added_records":len(added),"records":len(base)+len(added),"history_sha256":hsha,"output_sha256":sha(q),"training_started":False,"training_approval":False,"limitations":["User-authored topic remains narrow and may overfit.","Historical document has not had independent content review.","No general Chinese quality claim."]}
 (OUT/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");return result

if __name__=="__main__":print(json.dumps(run(),ensure_ascii=False,indent=2))
