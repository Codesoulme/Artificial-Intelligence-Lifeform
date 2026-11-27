"""Evaluate frozen intent/span candidates on reviewed natural dialogue mentions."""
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
from scripts.audit_spora_isolated_fast_weights import SOURCE, base
from scripts.audit_spora_isolated_name_routes import MentionHead, encode, locate
from scripts.audit_spora_route_position_control_20261127 import ControlledFeatures
from scripts.deepseek_spora_duconv_dev_identity_audit import MANIFEST_SHA, digest, reviewed_sample, select_sample
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_separate_intent_location_20261127 import IntentHead
from spora.research_ttt_ntp import attach_read_adapters, install

OUT = ROOT / 'runs/research/route-duconv-mentions-20261127-v1'
MATCHED = ROOT / 'runs/research/route-position-matched-training-20261127-v1'
PRIOR = ROOT / 'runs/research/separate-intent-location-20261127-v1'


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
                 'matched_report': MATCHED / 'result.json', 'default': MATCHED / 'default.pt',
                 'explicit': MATCHED / 'explicit.pt', 'prior': PRIOR / 'candidate.pt', 'prior_report': PRIOR / 'result.json'}
        before = {k: base.sha(v) for k, v in paths.items()}
        matched = json.loads(paths['matched_report'].read_text(encoding='utf8'))
        prior = json.loads(paths['prior_report'].read_text(encoding='utf8'))
        for scheme in ('default', 'explicit'):
            if before[scheme] != matched['arms'][scheme]['candidate_sha256']: raise ValueError('Candidate mismatch')
        if before['prior'] != prior['candidate_sha256'] or any(before[k] != matched['input_hashes'][k] for k in ('base', 'adapter')):
            raise ValueError('Source mismatch')
        sample = reviewed_sample(select_sample())
        examples = [{'text': r['text'], 'target': None} for r in sample]
        torch.set_num_threads(2); torch.cuda.reset_peak_memory_stats()
        stage('loading_model')
        adapter = torch.load(paths['adapter'], map_location='cpu', weights_only=True)
        model = load_full_streamed().eval()
        tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
        wrappers = install(model, tuple(adapter['layers']))
        for layer, reader in attach_read_adapters(wrappers, rank=adapter['rank']).items():
            reader.load_state_dict(adapter['adapters'][str(layer)], strict=True)
        model.requires_grad_(False)
        results = {}
        with torch.no_grad():
            for scheme in ('default', 'explicit', 'prior'):
                stage('encoding_' + scheme)
                payload = torch.load(paths[scheme], map_location='cpu', weights_only=True)
                locator = MentionHead(model.config.hidden_size).eval(); locator.load_state_dict(payload['locator'])
                intent = IntentHead(model.config.hidden_size).eval(); intent.load_state_dict(payload['intent'])
                extractor = ControlledFeatures(SimpleNamespace(base=model), tokenizer, depth=12, explicit_positions=(scheme == 'explicit'))
                data = encode(extractor, examples)
                hits = locate(locator, data, examples)
                probs = torch.sigmoid(intent(data[0], data[1])).tolist()
                rows = []
                for row, hit, p in zip(sample, hits, probs, strict=True):
                    subject, text = row['subject'], row['text']
                    representable = subject is not None and 2 <= len(subject) <= 4 and text.count(subject) == 1
                    passed = hit['margin'] > payload['locator_threshold']
                    rows.append({'text_sha256': digest(text), 'group': row['reviewed_group'],
                        'representable': representable, 'span_exact': subject is not None and hit['name'] == subject,
                        'margin_pass': passed, 'intent_pass': p >= payload['intent_threshold'],
                        'joint_pass': passed and p >= payload['intent_threshold'], 'margin': hit['margin'],
                        'intent_probability': p})
                summary = {}
                for group in ('single', 'none', 'multiple'):
                    items = [r for r in rows if r['group'] == group]
                    summary[group] = {'n': len(items), 'representable': sum(r['representable'] for r in items),
                        'top_exact': sum(r['span_exact'] for r in items),
                        'exact_margin_pass': sum(r['span_exact'] and r['margin_pass'] for r in items),
                        'margin_pass': sum(r['margin_pass'] for r in items),
                        'intent_pass': sum(r['intent_pass'] for r in items), 'joint_pass': sum(r['joint_pass'] for r in items)}
                results[scheme] = {'cutoff': payload['locator_threshold'], 'summary': summary, 'rows': rows}
        stage('verifying_inputs')
        after = {k: base.sha(v) for k, v in paths.items()}
        if before != after: raise ValueError('Protected input changed')
        output = {'arms': results, 'input_hashes': before, 'input_hashes_after': after,
                  'manifest_sha256': MANIFEST_SHA, 'seconds': time.perf_counter() - start,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        (OUT / 'result.json').write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        stage('complete')
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__': main()
