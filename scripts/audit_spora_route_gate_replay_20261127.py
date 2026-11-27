"""Reload a fixed routing candidate and attribute each rejected route to a gate."""
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
from scripts.audit_spora_route_contrast_20261127 import cases as old_cases
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_separate_intent_location_20261127 import IntentHead, FORMS, rows_for
from spora.identity_fingerprints import IdentityFingerprints
from spora.research_ttt_ntp import attach_read_adapters, install

PARENT = ROOT / 'runs/research/separate-intent-location-20261127-v1'
OUT = ROOT / 'runs/research/route-gate-replay-20261127-v1'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    def stage(name):
        item = {'stage': name, 'elapsed_seconds': time.perf_counter() - started}
        (OUT / 'stage.json').write_text(json.dumps(item), encoding='utf8')
        print(json.dumps(item), flush=True)
    try:
        stage('hashing')
        paths = {'base': base.MODEL / 'model.safetensors', 'adapter': SOURCE / 'adapters.pt',
                 'candidate': PARENT / 'candidate.pt', 'report': PARENT / 'result.json'}
        before = {k: base.sha(v) for k, v in paths.items()}
        report = json.loads(paths['report'].read_text(encoding='utf8'))
        if before['candidate'] != report['candidate_sha256']:
            raise ValueError('Candidate differs from first run')
        for k in ('base', 'adapter'):
            if before[k] != report['input_hashes'][k]:
                raise ValueError('Source differs from first run: ' + k)
        payload = torch.load(paths['candidate'], weights_only=True, map_location='cpu')
        if payload['locator_threshold'] != report['locator_cutoff'] or payload['intent_threshold'] != report['intent_cutoff']:
            raise ValueError('Saved thresholds differ')
        previous = old_cases()
        for i, row in enumerate(previous):
            name = NAMES[i // 5]
            row['target'] = name if row['group'] == 'read' else name + '宁' if row['group'] == 'unknown_longer' else None
            row['intent'] = int(row['target'] is not None)
            row['expected_key'] = row.pop('expected')
        new = rows_for(NAMES[:8], FORMS['new'], is_eval=True)
        stage('loading_model')
        torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        adapter = torch.load(paths['adapter'], weights_only=True, map_location='cpu')
        model = load_full_streamed().eval()
        tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
        wrappers = install(model, tuple(adapter['layers']))
        for layer, reader in attach_read_adapters(wrappers, rank=adapter['rank']).items():
            reader.load_state_dict(adapter['adapters'][str(layer)], strict=True)
        model.requires_grad_(False)
        locator = MentionHead(model.config.hidden_size).eval();locator.load_state_dict(payload['locator'])
        intent = IntentHead(model.config.hidden_size).eval();intent.load_state_dict(payload['intent'])
        bank = IdentityFingerprints(len(NAMES))
        for key, name in enumerate(NAMES):
            bank.bind(key, name)
        extractor = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer, depth=12)
        rows_out = []
        original_rows = {(r['set'], r['index']): r for r in report['rows']}
        with torch.no_grad():
            for name, examples in [('previous', previous), ('new', new)]:
                stage('encoding_' + name)
                data = encode(extractor, examples)
                found = locate(locator, data, examples)
                probabilities = torch.sigmoid(intent(data[0], data[1])).tolist()
                for i, (example, hit, probability) in enumerate(zip(examples, found, probabilities, strict=True)):
                    threshold_pass = hit['margin'] > payload['locator_threshold']
                    raw_key = route(bank, example['text'], hit, float('-inf'))
                    accepted_key = route(bank, example['text'], hit, payload['locator_threshold']) if probability >= payload['intent_threshold'] else None
                    prior = original_rows[(name, i)]
                    rows_out.append({'set': name, 'index': i, 'group': example['group'],
                        'expected_key': example['expected_key'], 'intent_probability': probability,
                        'intent_pass': probability >= payload['intent_threshold'], 'span_exact': hit['name'] == example['target'],
                        'span_margin': hit['margin'], 'threshold_pass': threshold_pass, 'raw_key': raw_key,
                        'selected_key': accepted_key, 'prior_selected_key': prior['new_selected'],
                        'matches_prior': accepted_key == prior['new_selected'],
                        'probability_delta': abs(probability - prior['intent_probability'])})
        stage('verifying_inputs')
        after = {k: base.sha(v) for k, v in paths.items()}
        result = {'rows': rows_out, 'input_hashes': before, 'input_hashes_after': after,
                  'locator_cutoff': payload['locator_threshold'], 'intent_cutoff': payload['intent_threshold'],
                  'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'limitations': ['Previously inspected synthetic development examples.',
                                  'Manual identity table, no memory reading or generation.',
                                  'Gate analysis does not establish threshold safety on new data.']}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        if before != after:
            raise ValueError('Protected input changed')
        stage('complete')
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__':
    main()
