"""Build a train-only Unicode vocabulary bound to a fixed manifest."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spora.text_data import read_manifest
from spora.codepoint import build_from_documents
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError(out)
 vocab=build_from_documents(read_manifest(a.manifest),'train');out.write_text(json.dumps(vocab.payload(),ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps({'vocab_size':vocab.size,'symbols':len(vocab.symbols)}))
if __name__=='__main__':main()
