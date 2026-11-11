"""Read-only fresh wording and near-name audit for the experimental binding.

Names and expected keys are scoring labels only. No user state is written.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import NAMES, KEYS, pooled, predict
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_spora_cluener_locator_retrain import OUTPUT as LOCATOR
from scripts.deepseek_spora_named_binding_retrain import OUTPUT as BINDING
from scripts.deepseek_spora_observation_transfer_audit import CHECKPOINT
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.named_address import address_basis_sha256
from spora.structure_transfer import SPORALayerReplacement

OUTPUT = ROOT / 'runs/deepseek-spora-name-address-stress-audit-20261111.json'
KNOWN_TEMPLATES = ('请查一下{name}记下的数字。', '{name}的记录目前是什么？',
                   '告诉我{name}之前留下的偏好。')
UNKNOWN_NAMES = ('小林林', '林小林', '小宁', '阿宁宁', '小羽', '阿哲哲', '梅小梅', '阿远远')
UNKNOWN_TEMPLATES = ('请查一下{name}记下的数字。', '{name}的记录目前是什么？')


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = (MODEL / 'model.safetensors', CHECKPOINT, LOCATOR / 'locator.pt', BINDING / 'binding.pt')
    hashes = {str(path): sha(path) for path in paths}
    if hashes[str(paths[0])] != WEIGHT_SHA:
        raise ValueError('Unexpected base weights')
    loc = torch.load(paths[2], map_location='cpu', weights_only=True)
    binding = torch.load(paths[3], map_location='cpu', weights_only=True)
    rows = [{'name': n, 'expected': k, 'text': t.format(name=n)}
            for n, k in zip(NAMES, KEYS) for t in KNOWN_TEMPLATES]
    rows += [{'name': n, 'expected': None, 'text': t.format(name=n)}
             for n in UNKNOWN_NAMES for t in UNKNOWN_TEMPLATES]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    torch.set_num_threads(2)
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    decoder, organism, state = load_integrated(CHECKPOINT, base, MODEL, individual_device='cuda')
    blocks = {i: layer for i, layer in enumerate(base.model.layers) if isinstance(layer, SPORALayerReplacement)}
    variant = replacement_feature_variant_sha256(MODEL, blocks)
    if any(data.get('feature_variant_sha256') != variant for data in (loc, binding)):
        raise ValueError('Feature variant mismatch')
    basis_hash = address_basis_sha256(organism)
    features, masks = extract(base, tokenizer, [row['text'] for row in rows], batch_size=2)
    token_vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
    locator = nn.Linear(loc['state']['weight'].shape[1], 1).eval()
    locator.load_state_dict(loc['state'], strict=True)
    with torch.inference_mode():
        scores = locator(token_vectors).squeeze(-1).sigmoid()
    spans, flags = [], []
    for index, score in enumerate(scores):
        mask, single = single_span_mask(score, masks[index])
        flags.append(single)
        spans.append(pooled(features[index:index + 1], mask.unsqueeze(0))[0])
    bases = torch.stack([F.normalize(site.current_basis.detach().float().cpu(), dim=-1) for site in organism.memories])
    indices, similarities = predict(torch.stack(spans), binding['weights'], bases)
    results = []
    for row, flag, index, similarity in zip(rows, flags, indices, similarities):
        predicted, values = index.tolist(), similarity.tolist()
        accepted = bool(flag and predicted[0] == predicted[1] and min(values) >= binding['threshold'])
        results.append({**row, 'single_span': flag, 'predicted': predicted, 'similarities': values,
                        'accepted': accepted, 'correct': bool(accepted and row['expected'] is not None
                                                               and predicted == [row['expected']] * 2)})
    known = [row for row in results if row['expected'] is not None]
    unknown = [row for row in results if row['expected'] is None]
    summary = {'known_total': len(known), 'known_correct': sum(row['correct'] for row in known),
               'known_rejected': sum(not row['accepted'] for row in known),
               'known_wrong_accepted': sum(row['accepted'] and not row['correct'] for row in known),
               'unknown_total': len(unknown), 'unknown_false_accept': sum(row['accepted'] for row in unknown)}
    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('Input artifact changed')
    report = {'status': 'development_only_no_writes', 'summary': summary, 'rows': results,
              'source_hashes': hashes, 'feature_variant_sha256': variant, 'address_basis_sha256': basis_hash,
              'threshold': binding['threshold'], 'seconds': time.perf_counter() - started,
              'limitations': ['Forty manually authored queries, small fixed set of names.',
                              'Single subject-span agreement is not proof that the subject is correct.',
                              'Expected labels are used only for metrics; no state or individual was modified.',
                              'The binding was trained on whole-question pooling; this audit uses predicted name spans.',
                              'No real chat, source trust, or write capability was evaluated.']}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
