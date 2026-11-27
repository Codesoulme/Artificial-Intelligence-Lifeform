"""Compare short and contextual synthetic training on the same frozen backbone."""
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
from scripts.audit_p236_duconv import ARCHIVE
from scripts.audit_spora_duconv_paired_intent_20261127 import READ, DECLINE
from scripts.audit_spora_isolated_fast_weights import NAMES, SOURCE, base
from scripts.audit_spora_isolated_name_routes import CAL_NAMES, TRAIN_NAMES, MentionHead, encode, locate, route
from scripts.audit_spora_route_disjoint_calibration_20261127 import FORMS as EVAL_FORMS, make_rows, summarize
from scripts.audit_spora_route_position_control_20261127 import ControlledFeatures
from scripts.deepseek_spora_duconv_dev_identity_audit import reviewed_sample, select_sample, digest
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.train_spora_route_position_matched_20261127 import choose, predict
from scripts.train_spora_separate_intent_location_20261127 import FORMS, IntentHead, rows_for
from spora.identity_fingerprints import IdentityFingerprints
from spora.research_ttt_ntp import attach_read_adapters, install

OUT = ROOT / 'runs/research/contextual-route-training-20261127-v1'
MATCHED = ROOT / 'runs/research/route-position-matched-training-20261127-v1'
PRIOR = ROOT / 'runs/research/duconv-paired-intent-20261127-v1/result.json'
CONTEXT = [
 ('read', '刚才{name}说想去散步。回想这位朋友以前常喝哪种饮料？'),
 ('read', '我听{name}聊过旅行。你能翻找此人曾提到的饮品偏好吗？'),
 ('unknown_longer', '刚才{u}说想去散步。回想这位朋友以前常喝哪种饮料？'),
 ('unknown_longer', '我听{u}聊过旅行。你能翻找此人曾提到的饮品偏好吗？'),
 ('excluded', '刚才{name}说想去散步。别回想那人的饮品喜好，算一加二。'),
 ('excluded', '我听{name}聊过旅行。不查询此人以前爱喝啥，请讲天气。'),
 ('word_task', '刚才{name}说想去散步。这个名字写了几个字？'),
 ('word_task', '我听{name}聊过旅行。把提到的姓名再写一遍。'),
 ('two_people', '刚才{name}与{b}说想去散步。回想他们各自常喝什么？'),
 ('two_people', '我听{name}和{b}聊过旅行。分别查一下两人的饮品喜好。')]


def train_rows():
    # rows_for formats {a}, not {name}; keep the forms and name splits explicit.
    forms = [(group, text.replace('{name}', '{a}')) for group, text in CONTEXT]
    return {'short': rows_for(TRAIN_NAMES, FORMS['train']), 'context': rows_for(TRAIN_NAMES, forms)}


def diagnostic_rows():
    sample = []
    for row in reviewed_sample(select_sample()):
        name = row['subject']
        if row['reviewed_group'] != 'single' or name == '文章' or name is None: continue
        if not (2 <= len(name) <= 3 and all('\u4e00' <= ch <= '\u9fff' for ch in name) and row['text'].count(name) == 1): continue
        sample.append(row)
    if len(sample) != 22 or {r['subject'] for r in sample} & (set(TRAIN_NAMES) | set(CAL_NAMES)):
        raise ValueError('Diagnostic sample differs or overlaps training people')
    rows = []
    for row in sample:
        name, text = row['subject'], row['text']
        for known in (True, False):
            current = text if known else text.replace(name, name + '宁', 1)
            for read in (True, False):
                rows.append({'text': current + (READ if read else DECLINE), 'target': None, 'name': name,
                             'mention': name if known else name + '宁', 'expected': 0 if known and read else None,
                             'group': ('known' if known else 'unknown') + ('_read' if read else '_decline'),
                             'source_sha256': digest(text)})
    return rows


def diagnose(head, intent, data, examples, cutoff):
    hits = locate(head, data, examples)
    probs = torch.sigmoid(intent(data[0], data[1])).tolist()
    rows = []
    for row, hit, probability in zip(examples, hits, probs, strict=True):
        bank = IdentityFingerprints(1); bank.bind(0, row['name'])
        ip, mp = probability >= .5, hit['margin'] > cutoff
        raw = route(bank, row['text'], hit, -1000000.)
        choice = raw if ip and mp else None
        rows.append({'group': row['group'], 'source_sha256': row['source_sha256'],
                     'expected': row['expected'], 'chosen': choice, 'intent_pass': ip, 'margin_pass': mp,
                     'span_exact': hit['name'] == row['mention'], 'raw_match': raw is not None})
    summary = {}
    for group in ('known_read', 'known_decline', 'unknown_read', 'unknown_decline'):
        items = [r for r in rows if r['group'] == group]
        summary[group] = {'n': len(items), 'correct': sum(r['chosen'] == r['expected'] for r in items),
            'false_reject': sum(r['expected'] == 0 and r['chosen'] is None for r in items),
            'false_activation': sum(r['expected'] is None and r['chosen'] is not None for r in items),
            'span_exact': sum(r['span_exact'] for r in items), 'intent_pass': sum(r['intent_pass'] for r in items),
            'margin_pass': sum(r['margin_pass'] for r in items)}
    return summary, rows


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
                 'prior': PRIOR, 'matched_report': MATCHED / 'result.json', 'prior_candidate': MATCHED / 'explicit.pt'}
        before = {k: base.sha(v) for k, v in paths.items()}
        matched = json.loads(paths['matched_report'].read_text(encoding='utf8'))
        if any(before[k] != matched['input_hashes'][k] for k in ('base', 'adapter')) or before['prior_candidate'] != matched['arms']['explicit']['candidate_sha256']:
            raise ValueError('Protected model source mismatch')
        train = train_rows()
        cal = make_rows(NAMES[:8], EVAL_FORMS['cal'])
        held = make_rows(NAMES[24:32], EVAL_FORMS['held'])
        diagnostics = diagnostic_rows()
        if len(train['short']) != 200 or len(train['context']) != 200: raise ValueError('Unequal training size')
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
            features = {name: encode(extractor, rows) for name, rows in
                        [('short', train['short']), ('context', train['context']), ('cal', cal), ('held', held), ('diagnostic', diagnostics)]}
        arms = {}
        for name in ('short', 'context', 'prior'):
            stage('training_' + name)
            torch.manual_seed(26301)
            head = MentionHead(model.config.hidden_size); intent = IntentHead(model.config.hidden_size)
            if name == 'prior':
                prior = torch.load(paths['prior_candidate'], map_location='cpu', weights_only=True)
                head.load_state_dict(prior['locator']); intent.load_state_dict(prior['intent'])
                cutoff = prior['locator_threshold']; valid = candidates = None
            else:
                optimizer = torch.optim.AdamW(list(head.parameters()) + list(intent.parameters()), lr=.002)
                generator = torch.Generator().manual_seed(26301)
                x, mask, targets = features[name]
                labels = torch.tensor([r['intent'] for r in train[name]], dtype=torch.float32)
                for epoch in range(20):
                    for batch in torch.randperm(len(x), generator=generator).split(32):
                        loss = F.cross_entropy(head(x[batch], mask[batch]), targets[batch]) + F.binary_cross_entropy_with_logits(intent(x[batch], mask[batch]), labels[batch])
                        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                head.eval(); intent.eval()
                with torch.no_grad(): cutoff, valid, candidates = choose(predict(head, intent, features['cal'], cal, bank))
                torch.save({'locator': head.state_dict(), 'intent': intent.state_dict(), 'locator_threshold': cutoff,
                            'intent_threshold': .5, 'training': name, 'seed': 26301}, OUT / (name + '.pt'))
            head.eval(); intent.eval()
            with torch.no_grad():
                synthetic = {k: predict(head, intent, features[k], rows, bank) for k, rows in [('cal', cal), ('held', held)]}
                summary, rows = diagnose(head, intent, features['diagnostic'], diagnostics, cutoff)
            arms[name] = {'cutoff': cutoff, 'valid_candidates': valid, 'candidate_count': candidates,
                          'synthetic_summary': {k: summarize(v, cutoff) for k, v in synthetic.items()},
                          'diagnostic_summary': summary, 'diagnostic_rows': rows,
                          'candidate_sha256': base.sha(OUT / (name + '.pt')) if name != 'prior' else before['prior_candidate']}
        stage('verifying_inputs')
        after = {k: base.sha(v) for k, v in paths.items()}
        if before != after: raise ValueError('Protected source changed')
        result = {'arms': arms, 'input_hashes': before, 'input_hashes_after': after,
                  'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        stage('complete')
    except BaseException:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise


if __name__ == '__main__': main()
