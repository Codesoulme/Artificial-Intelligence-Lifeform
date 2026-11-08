"""Audit recorded story positions against the exact immutable token stream."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tokenizers import Tokenizer

from spora.document_stream import DocumentStream
from spora.story_coverage_run import coverage_contract, restore_coverage


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan_path, checkpoint_path, out = ROOT / args.plan, ROOT / args.checkpoint, ROOT / args.output
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    contract = coverage_contract(plan)
    if not contract['record_story']:
        raise ValueError('Story position recording was not enabled')
    source_dir = ROOT / plan['data']
    manifest = json.loads((source_dir / 'manifest.json').read_text(encoding='utf8'))
    source, vocabulary = source_dir / 'pretrain-train.bin', source_dir / 'tokenizer.json'
    source_hash, tokenizer_hash = sha256(source), sha256(vocabulary)
    if source_hash != manifest['files'][source.name] or tokenizer_hash != manifest['tokenizer_sha256']:
        raise ValueError('Story stream or tokenizer differs from its manifest')
    before = sha256(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (payload.get('coverage_entry') != 'spora-story-coverage-v1'
            or payload.get('plan_sha256') != sha256(plan_path)
            or payload.get('manifest_sha256') != sha256(source_dir / 'manifest.json')
            or payload.get('story_source_sha256') != source_hash):
        raise ValueError('Checkpoint provenance mismatch')
    tokenizer = Tokenizer.from_file(str(vocabulary))
    stream = np.memmap(source, dtype=np.uint16, mode='r')
    documents = DocumentStream(stream, *(tokenizer.token_to_id(k) for k in ('<|bos|>', '<|eos|>', '<|pad|>')))
    coverage = restore_coverage(payload, contract, source_hash, documents).summaries()
    if sha256(checkpoint_path) != before:
        raise ValueError('Checkpoint changed during audit')
    total = sum(v.total_positions for v in coverage.values())
    unique = sum(v.unique_positions for v in coverage.values())
    result = {'scope': 'story positions within this run only; not unique across ancestors or sources',
              'status': 'audit_not_chat_acceptance', 'checkpoint_sha256': before,
              'plan_sha256': sha256(plan_path), 'source_sha256': source_hash,
              'tokenizer_sha256': tokenizer_hash, 'step': payload['steps'],
              'documents_seen': len(coverage), 'source_documents': len(documents.starts),
              'presented_positions': total, 'unique_positions': unique,
              'repeated_presentations': total - unique,
              'excluded': contract['excluded_sources'],
              'limits': ['Document indices depend on the exact source stream.',
                         'Checks count and bounds; it cannot reconstruct past batch contents.',
                         'No claim about ancestor exposure, source quality or semantic independence.']}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
