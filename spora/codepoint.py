"""Deterministic Unicode-character vocabulary and windows for local research."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
from .language_bridge import BOS,EOS,PAD,LanguageBridge

BASE=259
UNK=BASE

class CodepointVocabulary:
    def __init__(self, symbols):
        self.symbols=tuple(symbols)
        if len(set(self.symbols))!=len(self.symbols):raise ValueError('Duplicate vocabulary symbol')
        self.ids={char:BASE+1+i for i,char in enumerate(self.symbols)}
    @property
    def size(self):return BASE+1+len(self.symbols)
    def encode(self,text):
        if not isinstance(text,str):raise TypeError('Text required')
        return [BOS]+[self.ids.get(char,UNK) for char in text]+[EOS]
    def decode(self,tokens):
        reverse={v:k for k,v in self.ids.items()};return ''.join(reverse.get(int(x),'�') for x in tokens if int(x)>=BASE+1)
    def payload(self):return {'format':'spora-codepoint-vocab-v1','symbols':list(self.symbols),'vocab_size':self.size}
    @classmethod
    def from_file(cls,path):
        data=json.loads(Path(path).read_text(encoding='utf8'))
        if data.get('format')!='spora-codepoint-vocab-v1':raise ValueError('Invalid codepoint vocabulary')
        return cls(data.get('symbols',[]))

def build_from_documents(documents,split='train'):
    chars=set()
    for row in documents:
        if row['split']==split:chars.update(row['text'])
    return CodepointVocabulary(sorted(chars))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def windows(documents,vocab,split,length):
    if length<2:raise ValueError('Window too short')
    rows=[]
    for doc in documents:
        if doc['split']!=split:continue
        tokens=vocab.encode(doc['text'])
        for start in range(0,len(tokens)-1,length-1):
            piece=tokens[start:start+length];rows.append(piece+[PAD]*(length-len(piece)))
    if not rows:raise ValueError('No examples in selected split')
    return torch.tensor(rows,dtype=torch.long)

class CodepointLanguageBridge(LanguageBridge):
    def __init__(self,organism,vocab,context_layers=2,read_mode='exact'):
        self.vocabulary=vocab
        super().__init__(organism,read_mode=read_mode,context_layers=context_layers,vocab_size=vocab.size)
