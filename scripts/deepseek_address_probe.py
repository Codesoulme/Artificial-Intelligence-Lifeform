"""Inspect which input tokens a trained pooled address selector attends to."""
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


def run(checkpoint_path, output):
    checkpoint_path, output = Path(checkpoint_path).resolve(), Path(output).resolve()
    runs = (ROOT / 'runs').resolve()
    if not checkpoint_path.is_relative_to(runs) or not output.is_relative_to(runs) or output.exists():
        raise ValueError('Expected input and new output under runs')
    checkpoint_sha = sha(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') not in ('deepseek-spora-pooled-finite-address-v1',
                                          'deepseek-spora-question-pooled-finite-address-v1')
            or checkpoint.get('weight_sha256') != WEIGHT_SHA or checkpoint.get('individual_sha256') != INDIVIDUAL_SHA):
        raise ValueError('Checkpoint provenance mismatch')
    def verify():
        return (sha(MODEL / 'model.safetensors') == WEIGHT_SHA and sha(INDIVIDUAL) == INDIVIDUAL_SHA
                and sha(checkpoint_path) == checkpoint_sha)
    if not verify():
        raise ValueError('Protected inputs changed')
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
    questions = [('train', 26, '对象26现在的值是多少？'), ('held', 26, '对象26的值是多少？'),
                 ('novel', 26, '帮我查一下对象26的当前数值。'),
                 ('train', 20, '对象20现在的值是多少？'), ('held', 20, '对象20的值是多少？'),
                 ('novel', 20, '帮我查一下对象20的当前数值。')]
    features, masks = extract(model, tokenizer, [row[2] for row in questions],
                              question_only=checkpoint['format'] == 'deepseek-spora-question-pooled-finite-address-v1')
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': row[2]}], tokenize=False,
                                            add_generation_prompt=True) for row in questions]
    token_ids = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False)['input_ids']
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    rows = []
    with torch.inference_mode():
        for i, (category, key, question) in enumerate(questions):
            x, mask = features[i:i + 1].to('cuda').float(), masks[i:i + 1].to('cuda')
            addresses = selector(x, mask)
            sites = []
            for site in range(2):
                weights = selector.score[site](x[:, site]).squeeze(-1).masked_fill(~mask, float('-inf')).softmax(-1)[0]
                top = weights.topk(min(5, int(mask.sum()))).indices.tolist()
                similarities = addresses[site] @ bases[site].T
                sites.append({'predicted': int(similarities.argmax(-1)),
                              'correct_cosine': float(similarities[0, key]),
                              'top_tokens': [{'token': tokenizer.convert_ids_to_tokens(int(token_ids[i, j])),
                                              'weight': float(weights[j])} for j in top]})
            rows.append({'category': category, 'key': key, 'question': question, 'sites': sites})
    if not verify() or frozen_digest(organism, original) != before:
        raise ValueError('Protected inputs changed')
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump({'selector_sha256': checkpoint_sha, 'rows': rows,
                   'limitations': ['Six hand-selected inputs; attention weights are not a complete causal explanation.',
                                   'All entity/address pairs were supervised during training.']},
                  stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(rows, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
