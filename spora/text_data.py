"""Local, explicit, licensed-document manifest; no downloads or inferred consent."""
import hashlib,json
from pathlib import Path
import torch
from .language_bridge import encode,PAD

def read_manifest(path):
    path=Path(path).resolve();root=path.parent;manifest=json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema')!='spora-text-v1' or not isinstance(manifest.get('documents'),list):raise ValueError('Bad text manifest')
    seen=set();documents=[]
    for row in manifest['documents']:
        if set(row)!= {'path','sha256','split','source','license'}:raise ValueError('Document provenance fields required')
        if row['split'] not in ('train','development','test') or not row['source'] or not row['license']:
            raise ValueError('Valid split, source, and license required')
        document=(root/row['path']).resolve()
        if not document.is_relative_to(root):raise ValueError('Document must remain inside manifest directory')
        content=document.read_bytes();digest=hashlib.sha256(content).hexdigest()
        if digest!=row['sha256']:raise ValueError('Document hash mismatch')
        text=content.decode('utf-8-sig')
        normalized=' '.join(text.split())
        normalized_digest=hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        if normalized_digest in seen:raise ValueError('Duplicate document across or within splits')
        seen.add(normalized_digest)
        if not normalized:raise ValueError('Empty document')
        documents.append({**row,'text':text,'normalized_sha256':normalized_digest})
    if not all(any(r['split']==name for r in documents) for name in ('train','development','test')):
        raise ValueError('All three document-level splits required')
    return documents

def windows(documents,split,length=128):
    if length<2:raise ValueError('Window too short')
    result=[]
    for doc in documents:
        if doc['split']!=split:continue
        tokens=encode(doc['text'])
        for start in range(0,len(tokens)-1,length-1):
            piece=tokens[start:start+length]
            result.append(piece+[PAD]*(length-len(piece)))
    if not result:raise ValueError('No examples in selected split')
    return torch.tensor(result,dtype=torch.long)
