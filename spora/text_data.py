"""Local, explicit, licensed-document manifest; no downloads or inferred consent."""
import hashlib,json,unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import torch
from .language_bridge import encode,PAD

def read_manifest(path):
    path=Path(path).resolve();root=path.parent;manifest=json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema')!='spora-text-v1' or not isinstance(manifest.get('documents'),list):raise ValueError('Bad text manifest')
    if not 1 <= len(manifest['documents']) <= 4096:raise ValueError('Pilot manifest limited to 4096 documents')
    seen=set();documents=[];total_bytes=0
    for row in manifest['documents']:
        if set(row)!= {'path','sha256','split','source','license'}:raise ValueError('Document provenance fields required')
        if not all(isinstance(v,str) and v.strip() for v in row.values()):raise ValueError('Nonempty string provenance fields required')
        if row['split'] not in ('train','development','test') or not row['source'] or not row['license']:
            raise ValueError('Valid split, source, and license required')
        document=(root/row['path']).resolve()
        if not document.is_relative_to(root):raise ValueError('Document must remain inside manifest directory')
        total_bytes+=document.stat().st_size
        if total_bytes>4_000_000:raise ValueError('Pilot corpus limited to 4 MB; shard a larger reviewed corpus explicitly')
        content=document.read_bytes();digest=hashlib.sha256(content).hexdigest()
        if digest!=row['sha256']:raise ValueError('Document hash mismatch')
        text=content.decode('utf-8-sig')
        normalized=''.join(unicodedata.normalize('NFKC',text).casefold().split())
        normalized_digest=hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        if normalized_digest in seen:raise ValueError('Duplicate document across or within splits')
        seen.add(normalized_digest)
        if not normalized:raise ValueError('Empty document')
        documents.append({**row,'text':text,'normalized_sha256':normalized_digest})
    if not all(any(r['split']==name for r in documents) for name in ('train','development','test')):
        raise ValueError('All three document-level splits required')
    overlap_audit(documents)
    return documents


def overlap_audit(documents, width=13, minimum_shared=24, containment=.8):
    """Cross-split character shingles, including embedded copies.

    This bounded lexical filter is not semantic paraphrase detection or license
    verification. No word tokenizer is required for Chinese text.
    """
    index=defaultdict(list);sizes=[];examined=0;maximum=0.
    for number,doc in enumerate(documents):
        text=''.join(unicodedata.normalize('NFKC',doc['text']).casefold().split())
        signatures={hashlib.blake2b(text[i:i+width].encode('utf-8'),digest_size=8).digest()
                    for i in range(max(0,len(text)-width+1))}
        counts=Counter(other for sig in signatures for other in index[sig]
                       if documents[other]['split']!=doc['split'])
        for other,shared in counts.items():
            ratio=shared/min(len(signatures),sizes[other])
            examined+=1;maximum=max(maximum,ratio)
            if shared>=minimum_shared and ratio>=containment:
                raise ValueError('Cross-split near-duplicate: '+documents[other]['path']+' / '+doc['path'])
        sizes.append(len(signatures))
        for sig in signatures:index[sig].append(number)
    return {'shingle_characters':width,'minimum_shared':minimum_shared,'containment_threshold':containment,
            'candidate_pairs':examined,'maximum_containment':maximum,'passed':True}

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
