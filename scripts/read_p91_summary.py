"""Read a P91 summary only if it matches the current joint checkpoint."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_p91_paired import read_verified_summary

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
args = parser.parse_args()
print(json.dumps(read_verified_summary(args.directory), ensure_ascii=False, indent=2))
