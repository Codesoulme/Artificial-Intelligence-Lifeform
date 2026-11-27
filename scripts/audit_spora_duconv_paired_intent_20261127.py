"""Paired read/decline and known/near-name interventions on reviewed dialogue."""
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
from scripts.audit_spora_isolated_name_routes import MentionHead, encode, locate, route
from scripts.audit_spora_route_position_control_20261127 import ControlledFeatures
from scripts.deepseek_spora_duconv_dev_identity_audit import MANIFEST_SHA, digest, reviewed_sample, select_sample
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_separate_intent_location_20261127 import IntentHead
from spora.identity_fingerprints import IdentityFingerprints
from spora.research_ttt_ntp import attach_read_adapters, install

OUT = ROOT / 'runs/research/duconv-paired-intent-20261127-v1'
MATCHED = ROOT / 'runs/research/route-position-matched-training-20261127-v1'
READ = '你记得这个人以前喜欢喝什么饮料吗？'
DECLINE = '不用查这个人以前喜欢喝什么饮料，只回答二加三。'


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
                 'report': MATCHED / 'result.json', 'default': MATCHED / 'default.pt', 'explicit': MATCHED / 'explicit.pt'}
        before = {k: base.sha(v) for k, v in paths.items()}
        prior = json.loads(paths['report'].read_text(encoding='utf8'))
        if any(before[k] != prior['input_hashes'][k] for k in ('base', 'adapter')):
            raise ValueError('Base source mismatch')
        if any(before[k] != prior['arms'][k]['candidate_sha256'] for k in ('default', 'explicit')):
            raise ValueError('Candidate mismatch')
        sample = []
        for row in reviewed_sample(select_sample()):
            name = row['subject']
            if row['reviewed_group'] != 'single' or name == '文章' or name is None: continue
            if not (2 <= len(name) <= 3 and all('\u4e00' <= ch <= '\u9fff' for ch in name) and row['text'].count(name) == 1): continue
            sample.append(row)
        if not sample: raise ValueError('No suitable reviewed subjects')
        examples = []
        for row in sample:
            name = row['subject']
            unknown = name + '宁'
            if unknown in row['text'] or unknown == name: raise ValueError('Ambiguous generated near name')
            for known in (True, False):
                sentence = row['text'] if known else row['text'].replace(name, unknown, 1)
                for read in (True, False):
                    text = sentence + (READ if read else DECLINE)
                    examples.append({'text': text, 'target': None, 'name': name,
                                     'mention': name if known else unknown,
                                     'group': ('known' if known else 'unknown') + ('_read' if read else '_decline'),
                                     'expected': 0 if known and read else None,
                                     'source_sha256': digest(row['text'])})
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
            for scheme in ('default', 'explicit'):
                stage('encoding_' + scheme)
                payload = torch.load(paths[scheme], map_location='cpu', weights_only=True)
                locator = MentionHead(model.config.hidden_size).eval(); locator.load_state_dict(payload['locator'])
                intent = IntentHead(model.config.hidden_size).eval(); intent.load_state_dict(payload['intent'])
                extractor = ControlledFeatures(SimpleNamespace(base=model), tokenizer, depth=12, explicit_positions=(scheme == 'explicit'))
                data = encode(extractor, examples)
                hits = locate(locator, data, examples)
                probs = torch.sigmoid(intent(data[0], data[1])).tolist()
                rows = []
                for ex, hit, probability in zip(examples, hits, probs, strict=True):
                    bank = IdentityFingerprints(1); bank.bind(0, ex['name'])
                    margin_pass = hit['margin'] > payload['locator_threshold']
                    intent_pass = probability >= payload['intent_threshold']
                    raw = route(bank, ex['text'], hit, -1000000.)
                    choice = raw if margin_pass and intent_pass else None
                    rows.append({'source_sha256': ex['source_sha256'], 'group': ex['group'], 'expected': ex['expected'],
                        'chosen': choice, 'span_exact': hit['name'] == ex['mention'], 'margin_pass': margin_pass,
                        'intent_pass': intent_pass, 'raw_match': raw is not None})
                summary = {}
                for group in ('known_read', 'known_decline', 'unknown_read', 'unknown_decline'):
                    items = [r for r in rows if r['group'] == group]
                    summary[group] = {'n': len(items), 'correct': sum(r['chosen'] == r['expected'] for r in items),
                        'false_reject': sum(r['expected'] == 0 and r['chosen'] is None for r in items),
                        'false_activation': sum(r['expected'] is None and r['chosen'] is not None for r in items),
                        'span_exact': sum(r['span_exact'] for r in items), 'intent_pass': sum(r['intent_pass'] for r in items),
                        'margin_pass': sum(r['margin_pass'] for r in items), 'raw_match': sum(r['raw_match'] for r in items)}
                results[scheme] = {'summary': summary, 'rows': rows}
        stage('verifying_inputs')
        after = {k: base.sha(v) for k, v in paths.items()}
        if before != after: raise ValueError('Source changed')
        output = {'sample_count': len(sample), 'arms': results, 'manifest_sha256': MANIFEST_SHA,
                  'input_hashes': before, 'input_hashes_after': after, 'seconds': time.perf_counter() - start,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        (OUT / 'result.json').write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        stage('complete')
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__': main()
