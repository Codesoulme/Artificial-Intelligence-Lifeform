"""Cross the same registered names with two fixed previously examined phrasings."""
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_isolated_fast_weights import NAMES, SOURCE, base
from scripts.audit_spora_isolated_name_routes import MentionHead, encode, locate, route
from scripts.audit_spora_route_disjoint_calibration_20261127 import FORMS, make_rows, selected
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_separate_intent_location_20261127 import IntentHead
from spora.identity_fingerprints import IdentityFingerprints
from spora.research_ttt_ntp import attach_read_adapters, install

OUT = ROOT / 'runs/research/route-wording-crossover-20261127-v2'
PARENT = ROOT / 'runs/research/separate-intent-location-20261127-v1'
PRIOR = ROOT / 'runs/research/route-disjoint-calibration-20261127-v1/result.json'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    def stage(name):
        item = {'stage': name, 'seconds': time.perf_counter() - started}
        (OUT / 'stage.json').write_text(json.dumps(item), encoding='utf8')
        print(json.dumps(item), flush=True)
    try:
        stage('hashing')
        paths = {'base': base.MODEL / 'model.safetensors', 'adapter': SOURCE / 'adapters.pt',
                 'candidate': PARENT / 'candidate.pt', 'candidate_report': PARENT / 'result.json', 'prior': PRIOR}
        before = {k: base.sha(v) for k, v in paths.items()}
        old = json.loads(PRIOR.read_text(encoding='utf8'))
        if any(before[k] != old['input_hashes'][k] for k in ('base', 'adapter', 'candidate')):
            raise ValueError('Frozen sources differ from previous experiment')
        payload = torch.load(paths['candidate'], map_location='cpu', weights_only=True)
        names = NAMES[8:24]
        kinds = {'read', 'unknown_prefix', 'unknown_suffix'}
        examples = {label: [r for r in make_rows(names, forms) if r['group'] in kinds]
                    for label, forms in FORMS.items()}
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        stage('loading_model')
        adapter = torch.load(paths['adapter'], map_location='cpu', weights_only=True)
        model = load_full_streamed().eval()
        tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
        wrappers = install(model, tuple(adapter['layers']))
        for layer, reader in attach_read_adapters(wrappers, rank=adapter['rank']).items():
            reader.load_state_dict(adapter['adapters'][str(layer)], strict=True)
        model.requires_grad_(False)
        locator = MentionHead(model.config.hidden_size).eval(); locator.load_state_dict(payload['locator'])
        intent = IntentHead(model.config.hidden_size).eval(); intent.load_state_dict(payload['intent'])
        bank = IdentityFingerprints(len(NAMES))
        for i, name in enumerate(NAMES): bank.bind(i, name)
        extractor = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=12)
        out = {}
        with torch.no_grad():
            for label, rows in examples.items():
                stage('encoding_' + label)
                data = encode(extractor, rows)
                hits = locate(locator, data, rows)
                probabilities = torch.sigmoid(intent(data[0], data[1])).tolist()
                out[label] = [dict(group=row['group'], expected_key=row['expected_key'], text=row['text'],
                    raw_key=route(bank, row['text'], hit, -1000000.), margin=hit['margin'],
                    span=hit['name'], span_exact=hit['name'] == row['target'],
                    intent_probability=p, intent_pass=p >= .5)
                    for row, hit, p in zip(rows, hits, probabilities, strict=True)]
        prior_rows = {r['text']: r for group in old['rows'].values() for r in group}
        overlap = [(r, prior_rows[r['text']]) for rows in out.values() for r in rows if r['text'] in prior_rows]
        if len(overlap) != 48:
            raise ValueError('Unexpected previous overlap count')
        # Batching may alter floating point results very slightly; exact choices remain required.
        mismatch = [dict(text=a['text'], current=a, previous=b) for a, b in overlap
                    if a['span'] != b['span'] or a['raw_key'] != b['raw_key'] or a['intent_pass'] != b['intent_pass']
                    or abs(a['margin'] - b['margin']) > 1e-4]
        summary = {}
        for cutoff_name, cutoff in [('old', payload['locator_threshold']), ('calibrated', old['calibrated_cutoff'])]:
            summary[cutoff_name] = {}
            for group_label, section in [('prior_cal_people', slice(0, 8)), ('prior_held_people', slice(8, 16)), ('all', slice(0, 16))]:
                pairs = []
                for index in range(*section.indices(16)):
                    a, b = out['cal'][3 * index], out['held'][3 * index]
                    if a['group'] != 'read' or b['group'] != 'read' or a['expected_key'] != b['expected_key']:
                        raise ValueError('Incorrect paired case')
                    pairs.append((selected(a, cutoff) == a['expected_key'], selected(b, cutoff) == b['expected_key']))
                summary[cutoff_name][group_label] = {'both': pairs.count((True, True)),
                    'first_only': pairs.count((True, False)), 'second_only': pairs.count((False, True)),
                    'neither': pairs.count((False, False))}
            summary[cutoff_name]['unknown_activations'] = {label: {kind: sum(
                selected(r, cutoff) is not None for r in rows if r['group'] == kind)
                for kind in ('unknown_prefix', 'unknown_suffix')} for label, rows in out.items()}
        stage('verifying_inputs')
        after = {k: base.sha(v) for k, v in paths.items()}
        if before != after: raise ValueError('Protected input changed')
        result = {'summary': summary, 'rows': out, 'prior_overlap': len(overlap), 'prior_mismatch': mismatch, 'input_hashes': before,
                  'input_hashes_after': after, 'seconds': time.perf_counter() - started,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        stage('complete')
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__': main()
