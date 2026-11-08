"""Prepare a random Unicode-character interface; no text training."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from spora.growing import load
from spora.text_data import read_manifest
from spora.codepoint import CodepointVocabulary,CodepointLanguageBridge,sha
from spora.language_training import interface_state,load_interface
def main():
 p=argparse.ArgumentParser();p.add_argument('--individual',required=True);p.add_argument('--manifest',required=True);p.add_argument('--vocab',required=True);p.add_argument('--output',required=True);p.add_argument('--context-layers',type=int,default=2);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);read_manifest(a.manifest);vocab=CodepointVocabulary.from_file(a.vocab)
 parent=Path(a.individual);digest=sha(parent);organism,state,_=load(parent);bridge=CodepointLanguageBridge(organism,vocab,a.context_layers)
 payload={'format':'spora-codepoint-interface-v1','context_layers':a.context_layers,'vocab_size':vocab.size,'vocab_sha256':sha(a.vocab),'organism_sha256':digest,'state':interface_state(bridge),'trained_tokens':0}
 torch.save(payload,out/'untrained-interface.pt');restored=CodepointLanguageBridge(organism,vocab,a.context_layers);load_interface(restored,payload,digest)
 report={'status':'codepoint_interface_ready_not_trained','manifest_sha256':sha(a.manifest),'vocab_sha256':sha(a.vocab),'vocab_size':vocab.size,'symbols':len(vocab.symbols),'context_layers':a.context_layers,'training_started':False,'parent_unchanged':sha(parent)==digest}
 (out/'readiness.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
