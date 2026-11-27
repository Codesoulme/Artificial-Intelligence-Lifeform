"""Cross fixed intent and span heads without retraining or recalibration."""
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts.audit_p236_duconv import ARCHIVE
from scripts.audit_spora_isolated_fast_weights import NAMES, SOURCE, base
from scripts.audit_spora_isolated_name_routes import MentionHead, encode
from scripts.audit_spora_route_disjoint_calibration_20261127 import FORMS, make_rows, summarize
from scripts.audit_spora_route_position_control_20261127 import ControlledFeatures
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_contextual_route_20261127 import diagnostic_rows, diagnose
from scripts.train_spora_route_position_matched_20261127 import predict
from scripts.train_spora_separate_intent_location_20261127 import IntentHead
from spora.identity_fingerprints import IdentityFingerprints
from spora.research_ttt_ntp import attach_read_adapters, install

OUT = ROOT / 'runs/research/route-head-crossover-20261127-v1'
PARENT = ROOT / 'runs/research/contextual-route-training-20261127-v1'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    def stage(name):
        item = {'stage': name, 'seconds': time.perf_counter() - start}
        (OUT / 'stage.json').write_text(json.dumps(item), encoding='utf8')
        print(json.dumps(item), flush=True)
    try:
        stage('hashing')
        paths = {'base': base.MODEL / 'model.safetensors', 'adapter': SOURCE / 'adapters.pt', 'archive': ARCHIVE,
                 'parent_report': PARENT / 'result.json', 'short': PARENT / 'short.pt', 'context': PARENT / 'context.pt'}
        before = {k: base.sha(v) for k, v in paths.items()}
        previous = json.loads(paths['parent_report'].read_text(encoding='utf8'))
        if any(before[k] != previous['input_hashes'][k] for k in ('base', 'adapter', 'archive')):
            raise ValueError('Source differs from previous experiment')
        if any(before[k] != previous['arms'][k]['candidate_sha256'] for k in ('short', 'context')):
            raise ValueError('Candidate differs from previous experiment')
        cal, held = make_rows(NAMES[:8], FORMS['cal']), make_rows(NAMES[24:32], FORMS['held'])
        diagnostic = diagnostic_rows()
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
        extractor = ControlledFeatures(SimpleNamespace(base=model), tokenizer, depth=12, explicit_positions=True)
        stage('encoding')
        with torch.no_grad():
            data = {label: encode(extractor, rows) for label, rows in [('cal', cal), ('held', held), ('diagnostic', diagnostic)]}
            payloads = {k: torch.load(paths[k], map_location='cpu', weights_only=True) for k in ('short', 'context')}
            locators, intents = {}, {}
            for label, payload in payloads.items():
                locators[label] = MentionHead(model.config.hidden_size).eval()
                locators[label].load_state_dict(payload['locator'])
                intents[label] = IntentHead(model.config.hidden_size).eval()
                intents[label].load_state_dict(payload['intent'])
            arms = {}
            for locator_name in ('short', 'context'):
                for intent_name in ('short', 'context'):
                    stage('evaluating_' + locator_name + '_' + intent_name)
                    head, intent, cutoff = locators[locator_name], intents[intent_name], payloads[locator_name]['locator_threshold']
                    syn = {name: summarize(predict(head, intent, data[name], rows, bank), cutoff)
                           for name, rows in [('cal', cal), ('held', held)]}
                    diag, rows = diagnose(head, intent, data['diagnostic'], diagnostic, cutoff)
                    arms[locator_name + '_locator_' + intent_name + '_intent'] = {
                        'locator': locator_name, 'intent': intent_name, 'cutoff': cutoff,
                        'synthetic_summary': syn, 'diagnostic_summary': diag, 'diagnostic_rows': rows}
                    if locator_name == intent_name:
                        expected = previous['arms'][locator_name]
                        if syn != expected['synthetic_summary'] or diag != expected['diagnostic_summary']:
                            raise ValueError('Prior same-head result did not reproduce: ' + locator_name)
        stage('verifying_inputs')
        after = {k: base.sha(v) for k, v in paths.items()}
        if before != after: raise ValueError('Protected input changed')
        result = {'arms': arms, 'input_hashes': before, 'input_hashes_after': after,
                  'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        stage('complete')
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__': main()
