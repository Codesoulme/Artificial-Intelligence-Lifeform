"""Read-only order intervention on previously measured natural QA development items."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.audit_cmrc2018_dev_20261117 import SOURCE as DATA, BLOB_SHA1, blob_sha1
from scripts.cmrc_who_development_selection_20261117 import OUT as MANIFEST, select
from scripts.deepseek_cmrc_reading_train_pilot_20261117 import OUT as TRAINED
from scripts.deepseek_spora_general_subject_locator import extract, pack
from scripts.deepseek_spora_interleaved_natural_transaction import HEADS, SOURCE, source_paths
from scripts.deepseek_spora_natural_write_span_diagnostic import selected_text
from spora.research_context_span import ContextSpanSelector

OUT = ROOT / 'runs/deepseek-cmrc-question-first-probe-20261117.json'
PRIOR = ROOT / 'runs/deepseek-cmrc-reading-train-pilot-20261117/result.json'


def question_first(row):
    context, sep, question = row['text'].partition('\n问题：')
    if not sep or not 0 <= row['left'] < row['right'] <= len(context) or not question:
        raise ValueError('Invalid source row')
    prefix = '问题：' + question + '\n资料：'
    text = prefix + context
    left, right = row['left'] + len(prefix), row['right'] + len(prefix)
    if text[left:right] != context[row['left']:row['right']]:
        raise ValueError('Offset mismatch')
    return {**row, 'text': text, 'left': left, 'right': right}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {**source_paths(), 'data': DATA, 'manifest': MANIFEST,
             'candidate': TRAINED / 'head.pt', 'prior_report': PRIOR}
    hashes = {k: runtime.sha(p) for k, p in paths.items()}
    raw = DATA.read_bytes()
    if blob_sha1(raw) != BLOB_SHA1 or hashes['base'] != runtime.WEIGHT_SHA:
        raise ValueError('Source mismatch')
    rows, _ = select(json.loads(raw.decode('utf8')))
    manifest = json.loads(MANIFEST.read_text(encoding='utf8'))
    report = json.loads(PRIOR.read_text(encoding='utf8'))
    payload = torch.load(paths['candidate'], map_location='cpu', weights_only=True)
    if (manifest['source_sha256'] != hashes['data']
            or [r['digest'] for r in rows] != [r['digest'] for r in manifest['rows']]
            or hashes['candidate'] != report['head_sha256']
            or any(payload['source_sha256'][k] != hashes[k] for k in source_paths())):
        raise ValueError('Selection or candidate mismatch')
    transformed = [question_first(row) for row in rows]
    runtime.CHECKPOINT = SOURCE
    with tempfile.TemporaryDirectory(prefix='spora-question-first-') as folder:
        runtime.WORKING_CHECKPOINT = Path(folder) / 'not-created.pt'
        chat = runtime.load_chat(natural_write_heads_path=HEADS)
        items = []
        for row in transformed:
            items.extend(extract(chat, [row]))
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Unexpected individual file')
    head = ContextSpanSelector(items[0]['features'].shape[-1], rank=payload['rank']).eval()
    head.load_state_dict(payload['state'], strict=True)
    details = []
    for item in items:
        x, mask, target, usable = pack([item])
        with torch.inference_mode():
            choice = int(head(x, mask).argmax())
        selected = torch.zeros(len(item['mask']), dtype=torch.bool)
        if choice:
            left, right = divmod(choice - 1, x.shape[1])
            if right >= len(selected):
                raise ValueError('Padding selected')
            selected[left:right + 1] = True
        text = selected_text(item['row']['text'], selected, item['offsets'], item['start']) if bool(selected.any()) else None
        expected = item['row']['text'][item['row']['left']:item['row']['right']]
        details.append({'digest': item['row']['digest'], 'representable': bool(usable[0]),
                        'selected': text is not None, 'exact_text': text == expected,
                        'exact_position': choice != 0 and choice == int(target[0]) and bool(usable[0])})
    summary = {'cases': len(details), 'representable': sum(r['representable'] for r in details),
               'selected': sum(r['selected'] for r in details), 'exact_text': sum(r['exact_text'] for r in details),
               'exact_position': sum(r['exact_position'] for r in details),
               'previous_question_last': report['after']['dev']['known']}
    if any(runtime.sha(p) != hashes[k] for k, p in paths.items()):
        raise ValueError('Source changed')
    result = {'summary': summary, 'rows': details, 'source_sha256': hashes,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Question-first is out of distribution for the head and changes prompt formatting.',
                  'Previously inspected development cases; no independent or causal isolation of position effects.',
                  'Full paragraph is supplied; no memory, identity, unanswerable or generation test.',
                  'Known-answer task only; model answer spans do not establish personal identity.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
