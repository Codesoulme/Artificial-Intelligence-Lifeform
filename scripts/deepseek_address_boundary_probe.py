"""Inspect address confidence on valid and unsupported questions without a rejection policy."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_pooled_address import PooledAddress, extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest

CASES = (
    ('current', '对象0的值是多少？', 0), ('current', '对象8的值是多少？', 8),
    ('current', '对象26的值是多少？', 26), ('current', '对象63的值是多少？', 63),
    ('out_of_range', '对象64的值是多少？', None), ('out_of_range', '对象99的值是多少？', None),
    ('unrelated', '今天天气怎么样？', None), ('unrelated', '请介绍你自己。', None),
    ('history', '对象26昨天的值是多少？', None), ('history', '对象8上一次的值是多少？', None),
    ('multiple', '对象8和对象9的值之和是多少？', None),
    ('write', '请把对象26的值改成3。', None), ('malformed', '对象的值是多少？', None),
)


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    runs = (ROOT / 'runs').resolve()
    if not source.is_relative_to(runs) or not output.is_relative_to(runs) or output.exists():
        raise ValueError('Expected source and new output under runs')
    source_sha = sha(source)
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != 'deepseek-spora-question-pooled-finite-address-v1'
            or checkpoint.get('weight_sha256') != WEIGHT_SHA or checkpoint.get('individual_sha256') != INDIVIDUAL_SHA):
        raise ValueError('Source provenance mismatch')
    def verify():
        return (sha(MODEL / 'model.safetensors') == WEIGHT_SHA and sha(INDIVIDUAL) == INDIVIDUAL_SHA
                and sha(source) == source_sha)
    if not verify():
        raise ValueError('Protected source mismatch')
    torch.set_num_threads(2)
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    selector = PooledAddress(model.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    selector.load_state_dict(checkpoint['selector'], strict=True)
    features, masks = extract(model, tokenizer, [case[1] for case in CASES])
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    rows = []
    with torch.inference_mode():
        for i, (category, question, expected) in enumerate(CASES):
            addresses = selector(features[i:i + 1].to('cuda'), masks[i:i + 1].to('cuda'))
            sites = []
            for site in range(2):
                scores = (addresses[site] @ bases[site].T)[0]
                values, indices = scores.topk(2)
                sites.append({'predicted': int(indices[0]), 'top_cosine': float(values[0]),
                              'margin': float(values[0] - values[1])})
            rows.append({'category': category, 'question': question, 'expected_entity': expected, 'sites': sites})
    if not verify() or frozen_digest(organism, original) != before:
        raise ValueError('Protected source changed')
    report = {'status': 'address_boundary_diagnostic_no_rejection', 'selector_sha256': source_sha, 'rows': rows,
              'limitations': ['Small hand-selected sample; not a calibrated rejection evaluation.',
                              'All inputs are forced to choose one of 64 addresses.',
                              'High cosine or margin alone is not evidence a question is supported.',
                              'No memory value is read and no answer is generated.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(rows, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
