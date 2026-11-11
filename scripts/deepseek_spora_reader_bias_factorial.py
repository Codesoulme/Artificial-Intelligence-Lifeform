"""Intervene separately on decoder memory readers and output bias."""
from contextlib import ExitStack
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_counterfactual_output_audit import PERSON_INDICES, VALUES, TEMPLATES, OUTPUT as PRIOR
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_output_recalibration_pilot import make_state
from spora.growing import query
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputLogitsProcessor

OUTPUT = ROOT / 'runs/deepseek-spora-reader-bias-factorial-20261112.json'
FRESH_OUTPUT = ROOT / 'runs/deepseek-spora-reader-bias-fresh-20261112.json'
FRESH_TEMPLATES = ('你记得{name}目前喜欢喝什么饮料吗？', '{name}眼下的饮料偏好是什么？')
MODES = (('both', True, True), ('reader_only', True, False),
         ('bias_only', False, True), ('neither', False, False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    output_path = FRESH_OUTPUT if args.fresh else OUTPUT
    persons = (0, 1, 2, 3) if args.fresh else PERSON_INDICES
    templates = FRESH_TEMPLATES if args.fresh else TEMPLATES
    modes = tuple(m for m in MODES if not args.fresh or m[0] != 'bias_only')
    if output_path.exists():
        raise FileExistsError(output_path)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    previous = json.loads(PRIOR.read_text(encoding='utf-8'))
    if previous['source_checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Prior checkpoint mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        runtime.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: load_full_streamed()
        chat = runtime.load_chat()
        device = next(chat.organism.parameters()).device
        rows = []
        for person in persons:
            key, name, _ = ENROLLMENTS[person]
            for value in VALUES:
                chat.state = make_state(chat, person, value)
                before = frozen_digest(chat.organism, chat.state)
                keys = torch.tensor([key], dtype=torch.long, device=device)
                with torch.no_grad():
                    native_logits = query(chat.organism, chat.state, keys, torch.zeros_like(keys), torch.zeros_like(keys))
                memory_top = int(native_logits.argmax(-1)[0])
                for template_index, template in enumerate(templates):
                    question = template.format(name=name)
                    row = {'person': person, 'value': value, 'template': template_index,
                           'memory_top': memory_top, 'expected_word': WORDS[value], 'modes': {}}
                    for mode, reader_on, bias_on in modes:
                        with ExitStack() as stack:
                            if reader_on:
                                reader = stack.enter_context(patch.object(chat.decoder.reader, 'forward',
                                                                          wraps=chat.decoder.reader.forward))
                            else:
                                reader = stack.enter_context(patch.object(chat.decoder.reader, 'forward',
                                                                          side_effect=lambda hidden, *args, **kwargs: hidden))
                            if not bias_on:
                                stack.enter_context(patch.object(MemoryOutputLogitsProcessor, '__call__',
                                                                 lambda self, ids, scores: scores))
                            result = chat.answer(question)
                        row['modes'][mode] = {'result': result, 'reader_calls': reader.call_count,
                                              'correct': bool(result['accepted'] and result.get('key') == key
                                                              and result.get('answer') == WORDS[value])}
                        if result['accepted'] and reader.call_count == 0:
                            raise ValueError('Expected wrapped reader intervention was not exercised')
                    rows.append(row)
                    print(json.dumps({'person': person, 'value': value, 'template': template_index,
                                      **{mode: row['modes'][mode]['correct'] for mode, _, _ in modes}}), flush=True)
                if frozen_digest(chat.organism, chat.state) != before:
                    raise ValueError('Inference modified temporary individual')
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected checkpoint changed')
    summary = {'examples': len(rows), 'memory_top_matches': sum(r['memory_top'] == r['value'] for r in rows),
               **{mode + '_correct': sum(r['modes'][mode]['correct'] for r in rows) for mode, _, _ in modes},
               **{mode + '_accepted': sum(r['modes'][mode]['result']['accepted'] for r in rows) for mode, _, _ in modes}}
    counterfactuals = []
    for person in persons:
        for template_index in range(len(templates)):
            subset = [r for r in rows if r['person'] == person and r['template'] == template_index]
            counterfactuals.append({'person': person, 'template': template_index,
                                    **{mode + '_distinct_answers': len({r['modes'][mode]['result'].get('answer') for r in subset})
                                       for mode, _, _ in modes}})
    report = {'status': 'controlled_reader_bias_factorial', 'summary': summary, 'counterfactuals': counterfactuals,
              'rows': rows, 'source_checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter() - started,
              'limitations': [('Twenty-four authored conditions: four known names, two new templates and three finite values.'
                               if args.fresh else 'Twelve authored conditions from earlier development; two names and three finite values.'),
                              'Reader-off returns the layer hidden input unchanged; memory state is still checked for routing.',
                              'Bias-off suppresses the vocabulary processor but leaves other computation unchanged.',
                              'Manual numerical writes and exact single-word scoring, not natural conversation.',
                              'Two reader sites intervened together; does not establish contribution of each site.',
                              'No deployment, real individual changes, or generalization claims.']}
    with output_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'summary': summary, 'counterfactuals': counterfactuals}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
