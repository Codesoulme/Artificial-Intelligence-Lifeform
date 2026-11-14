"""Compare an isolated original-layer reader candidate on prior failures."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_original_reader_output_factorial import CASES

SOURCE = ROOT / 'runs/deepseek-spora-original-reader-mixed-20261114'
OUTPUT = ROOT / 'runs/deepseek-spora-original-reader-mixed-regression-20261114.json'
ORDINARY = ('请写一句鼓励学习的话。', '现在几点了？')


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha, checkpoint_sha = p.sha(p.MODEL / 'model.safetensors'), p.sha(p.runtime.CHECKPOINT)
    source_report = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    candidate_sha = p.sha(SOURCE / 'reader.pt')
    if base_sha != p.WEIGHT_SHA or candidate_sha != source_report['candidate_sha256']:
        raise ValueError('Protected artifact checksum mismatch')
    candidate = torch.load(SOURCE / 'reader.pt', map_location='cpu', weights_only=True)
    if (candidate['base_sha256'] != base_sha or candidate['source_checkpoint_sha256'] != checkpoint_sha
            or candidate['original_layers'] is not True):
        raise ValueError('Candidate provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    ordinary = []
    with tempfile.TemporaryDirectory(prefix='spora-original-mixed-regression-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        base = p.load_full_streamed().eval()
        originals = {i: base.model.layers[i] for i in (6, 12)}
        chat = p.runtime.load_chat(base_loader=lambda: base)
        source_digest = p.frozen_digest(chat.organism, chat.state)
        replacements = {i: base.model.layers[i] for i in originals}
        for i in originals:
            base.model.layers[i] = originals[i]
        baseline = {k: v.detach().cpu().clone() for k, v in chat.decoder.reader.state_dict().items()}
        weights = {'baseline': baseline, 'candidate': candidate['state']}

        def generate(question, args):
            prefix = closed_think_inputs(chat.tokenizer, question, 'cuda')
            with torch.inference_mode():
                ids = chat.decoder.generate(**prefix, **args, max_new_tokens=24, do_sample=False,
                                            pad_token_id=chat.tokenizer.pad_token_id,
                                            eos_token_id=chat.tokenizer.eos_token_id)
            tokens = ids[0, prefix['input_ids'].shape[1]:].tolist()
            return {'answer': chat.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                    'token_ids': tokens, 'ended': chat.tokenizer.eos_token_id in tokens}

        for name, key, question, first, second in CASES:
            for value in (first, second):
                state = p.fresh_controlled_state(chat.organism)
                state, decision = p.observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Controlled write rejected')
                digest = p.frozen_digest(chat.organism, state)
                args, evidence = p.runtime._prepare_memory_read(chat.organism, state, key, 'current', None)
                if int(evidence.argmax(-1)[0]) != value:
                    raise ValueError('Unexpected native value')
                answers = {}
                for label, weight in weights.items():
                    chat.decoder.reader.load_state_dict(weight, strict=True)
                    result = generate(question, args)
                    result['exact'] = result['answer'].rstrip('。.!！ ') == p.WORDS[value]
                    answers[label] = result
                rows.append({'name': name, 'question': question, 'key': key, 'value': p.WORDS[value],
                             'native_value_id': value, 'answers': answers})
                print(json.dumps(rows[-1], ensure_ascii=True), flush=True)
                if p.frozen_digest(chat.organism, state) != digest:
                    raise ValueError('Temporary state changed')
        state = p.fresh_controlled_state(chat.organism)
        state, decision = p.observe_sourced(chat.organism, state, 22, 2, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Control write rejected')
        args, _ = p.runtime._prepare_memory_read(chat.organism, state, 22, 'current', None)
        digest = p.frozen_digest(chat.organism, state)
        for question in ORDINARY:
            variants = {}
            for label, weight in weights.items():
                chat.decoder.reader.load_state_dict(weight, strict=True)
                variants[label] = {'without_memory': generate(question, {}), 'with_forced_memory': generate(question, args)}
            ordinary.append({'question': question, 'results': variants})
        if p.frozen_digest(chat.organism, state) != digest:
            raise ValueError('Control state changed')
        chat.decoder.reader.load_state_dict(baseline, strict=True)
        for i in originals:
            base.model.layers[i] = replacements[i]
        if p.frozen_digest(chat.organism, chat.state) != source_digest:
            raise ValueError('Source individual changed')
    if (p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha
            or p.sha(SOURCE / 'reader.pt') != candidate_sha):
        raise ValueError('Protected file changed')
    summary = {'conditions': len(rows), **{label + '_exact': sum(r['answers'][label]['exact'] for r in rows)
                                          for label in weights},
               **{label + '_paired_correct': sum(all(r['answers'][label]['exact'] for r in rows if r['name'] == name)
                                                 for name, _, _, _, _ in CASES) for label in weights}}
    report = {'summary': summary, 'rows': rows, 'ordinary': ordinary, 'base_sha256': base_sha,
              'checkpoint_sha256': checkpoint_sha, 'candidate_sha256': candidate_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Prior development prompts, not independent validation.',
                              'Manual numeric addresses and writes; no routing or natural learning.',
                              'Output adapter disabled; exact short answer is not general chat.',
                              'Ordinary requests with forced memory are stress tests, not actual router behavior.',
                              'Only original layers and temporary states; candidate remains isolated.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
