"""Build a character vocabulary using only the training split."""
import argparse
import json
from pathlib import Path

from spora.codepoint import build_from_documents
from spora.train_hybrid import read_non_test_documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    docs = read_non_test_documents(args.manifest)
    payload = build_from_documents(docs).payload()
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'symbols': len(payload['symbols']), 'test_split_read': False}))


if __name__ == '__main__':
    main()
