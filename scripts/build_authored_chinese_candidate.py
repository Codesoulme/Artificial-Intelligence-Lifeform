"""Build a provenance-bound candidate from user-authored/local reviewed text."""
from __future__ import annotations
import hashlib,json,re,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
DOCX=Path(r"E:\人工智能生命体\SPORA 架构：面向持续学习与生态演化的智能体系统设计.docx")
PASTED=Path(r"C:\Users\14262\.codex\attachments\5bc28f0f-1f20-4858-8345-863dfb449f37\已粘贴的文本.txt")
P35=ROOT/"runs/p35-authored-dialogue-v4/teacher.jsonl"
OUT=ROOT/"runs/p136-authored-chinese-candidate-20261109"

NS={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def docx_paragraphs(path):
 with zipfile.ZipFile(path) as z: root=ET.fromstring(z.read("word/document.xml"))
 out=[]
 for p in root.findall(".//w:p",NS):
  text="".join(t.text or "" for t in p.findall(".//w:t",NS)).strip()
  if text:out.append(text)
 return out

def run():
 if OUT.exists():raise FileExistsError(OUT)
 for p in (DOCX,PASTED,P35):
  if not p.exists():raise FileNotFoundError(p)
 records=[]; sources=[]
 for kind,path,paragraphs in [("user_architecture_docx",DOCX,docx_paragraphs(DOCX)),("user_context_text",PASTED,PASTED.read_text(encoding="utf-8"))]:
  texts=paragraphs if isinstance(paragraphs,list) else [x.strip() for x in re.split(r"\n{2,}",paragraphs) if x.strip()]
  for i,text in enumerate(texts):
   if len(text)<20:continue
   records.append({"record_id":f"{kind}:{i}","kind":kind,"source_file":str(path),"source_file_sha256":sha(path),"text":text,"review_status":"user_authored_not_quality_reviewed"})
  sources.append({"kind":kind,"path":str(path),"sha256":sha(path),"records":len(texts)})
 with P35.open(encoding="utf-8") as f:
  for i,line in enumerate(f):
   row=json.loads(line); text=row.get("text","")
   if text:records.append({"record_id":f"p35_teacher:{i}","kind":"reviewed_authored_dialogue","source_file":str(P35),"source_file_sha256":sha(P35),"text":text,"group":row.get("group"),"source":row.get("source"),"review_status":"p35_approved_for_pilot_only"})
 sources.append({"kind":"reviewed_authored_dialogue","path":str(P35),"sha256":sha(P35),"records":sum(1 for _ in P35.open(encoding="utf-8"))})
 OUT.mkdir(parents=False); q=OUT/"candidate.jsonl"
 with q.open("x",encoding="utf-8",newline="\n") as f:
  for r in records:f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n")
 result={"status":"candidate_not_trained_authored_sources","records":len(records),"source_files":sources,"output_sha256":sha(q),"contains_text":True,"training_started":False,"review_approved":False,"limitations":["Architecture/context text is narrow and can overfit topic and phrasing.","P35 approval is bounded pilot approval, not broad chat approval.","No external rights or semantic generalization claim; no training performed."]}
 (OUT/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 return result

if __name__=="__main__":print(json.dumps(run(),ensure_ascii=False,indent=2))
