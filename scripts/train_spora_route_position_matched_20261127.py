"""Matched retraining of intent and span heads under two position schemes."""
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch.nn import functional as F
from scripts.audit_spora_isolated_fast_weights import NAMES, SOURCE, base
from scripts.audit_spora_isolated_name_routes import CAL_NAMES, TRAIN_NAMES, MentionHead, encode, locate, route
from scripts.audit_spora_route_disjoint_calibration_20261127 import FORMS as EVAL_FORMS, make_rows, selected, summarize
from scripts.audit_spora_route_position_control_20261127 import ControlledFeatures
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_separate_intent_location_20261127 import FORMS, IntentHead, rows_for
from spora.identity_fingerprints import IdentityFingerprints
from spora.research_ttt_ntp import attach_read_adapters, install

OUT = ROOT / 'runs/research/route-position-matched-training-20261127-v1'
PARENT = ROOT / 'runs/research/separate-intent-location-20261127-v1'
SEED = 26301


def predict(head, intent, data, rows, bank):
    hits = locate(head, data, rows)
    probs = torch.sigmoid(intent(data[0], data[1])).tolist()
    return [dict(group=row['group'], text=row['text'], expected_key=row['expected_key'],
                 raw_key=route(bank, row['text'], hit, -1000000.), margin=hit['margin'],
                 intent_pass=p >= .5, intent_probability=p, span_exact=hit['name'] == row['target'],
                 span=hit['name']) for row, hit, p in zip(rows, hits, probs, strict=True)]


def choose(rows):
    candidates = sorted({-1000000., *(r['margin'] for r in rows)})
    options = []
    for cutoff in candidates:
        counts = summarize(rows, cutoff)
        if sum(v['wrong_address'] for v in counts.values()) == 0:
            options.append((sum(v['correct_read'] for v in counts.values()), cutoff))
    if not options: raise ValueError('No zero-error calibration threshold')
    return max(options)[1], len(options), len(candidates)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    def stage(name):
        v = {'stage': name, 'seconds': time.perf_counter() - start}
        (OUT / 'stage.json').write_text(json.dumps(v), encoding='utf8')
        print(json.dumps(v), flush=True)
    try:
        stage('hashing')
        paths = {'base': base.MODEL / 'model.safetensors', 'adapter': SOURCE / 'adapters.pt',
                 'prior_candidate': PARENT / 'candidate.pt', 'prior_report': PARENT / 'result.json'}
        before = {k: base.sha(v) for k, v in paths.items()}
        previous = json.loads(paths['prior_report'].read_text(encoding='utf8'))
        if before['prior_candidate'] != previous['candidate_sha256'] or any(before[k] != previous['input_hashes'][k] for k in ('base', 'adapter')):
            raise ValueError('Source mismatch')
        cal_names, held_names = NAMES[:8], NAMES[24:32]
        if set(cal_names) & set(held_names) or (set(cal_names) | set(held_names)) & (set(TRAIN_NAMES) | set(CAL_NAMES)):
            raise ValueError('People overlap')
        train = rows_for(TRAIN_NAMES, FORMS['train'])
        groups = {'cal': make_rows(cal_names, EVAL_FORMS['cal']), 'held': make_rows(held_names, EVAL_FORMS['held'])}
        torch.set_num_threads(2); torch.cuda.reset_peak_memory_stats()
        stage('loading_model')
        adapter = torch.load(paths['adapter'], map_location='cpu', weights_only=True)
        model = load_full_streamed().eval()
        tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
        wrappers = install(model, tuple(adapter['layers']))
        for layer, reader in attach_read_adapters(wrappers, rank=adapter['rank']).items():
            reader.load_state_dict(adapter['adapters'][str(layer)], strict=True)
        model.requires_grad_(False)
        bank = IdentityFingerprints(len(NAMES))
        for i, name in enumerate(NAMES): bank.bind(i, name)
        output = {}
        with torch.no_grad():
            for scheme, flag in [('default', False), ('explicit', True)]:
                extractor = ControlledFeatures(SimpleNamespace(base=model), tokenizer, depth=12, explicit_positions=flag)
                stage('encoding_' + scheme)
                data = {label: encode(extractor, rows) for label, rows in [('train', train), *groups.items()]}
                torch.manual_seed(SEED)
                head = MentionHead(model.config.hidden_size)
                intent = IntentHead(model.config.hidden_size)
                with torch.enable_grad():
                    optimizer = torch.optim.AdamW(list(head.parameters()) + list(intent.parameters()), lr=.002)
                    generator = torch.Generator().manual_seed(SEED)
                    x, mask, targets = data['train']
                    labels = torch.tensor([r['intent'] for r in train], dtype=torch.float32)
                    for epoch in range(20):
                        for batch in torch.randperm(len(train), generator=generator).split(32):
                            loss = F.cross_entropy(head(x[batch], mask[batch]), targets[batch]) + F.binary_cross_entropy_with_logits(intent(x[batch], mask[batch]), labels[batch])
                            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                head.eval(); intent.eval()
                cal = predict(head, intent, data['cal'], groups['cal'], bank)
                cutoff, valid_count, candidate_count = choose(cal)
                held = predict(head, intent, data['held'], groups['held'], bank)
                candidate_path = OUT / (scheme + '.pt')
                torch.save({'locator': head.state_dict(), 'intent': intent.state_dict(), 'locator_threshold': cutoff,
                            'intent_threshold': .5, 'seed': SEED, 'scheme': scheme}, candidate_path)
                output[scheme] = {'cutoff': cutoff, 'valid_candidates': valid_count, 'candidate_count': candidate_count,
                    'summary': {'cal': summarize(cal, cutoff), 'held': summarize(held, cutoff)},
                    'rows': {'cal': cal, 'held': held}, 'candidate_sha256': base.sha(candidate_path)}
                if scheme == 'default':
                    old = torch.load(paths['prior_candidate'], map_location='cpu', weights_only=True)
                    head.load_state_dict(old['locator']); intent.load_state_dict(old['intent'])
                    old_rows = {k: predict(head, intent, data[k], groups[k], bank) for k in groups}
                    output['prior'] = {'cutoff': old['locator_threshold'],
                        'summary': {k: summarize(rows, old['locator_threshold']) for k, rows in old_rows.items()}, 'rows': old_rows}
        stage('verifying_inputs')
        after = {k: base.sha(v) for k, v in paths.items()}
        if before != after: raise ValueError('Protected input changed')
        result = {'arms': output, 'input_hashes': before, 'input_hashes_after': after,
                  'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        stage('complete')
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__': main()
