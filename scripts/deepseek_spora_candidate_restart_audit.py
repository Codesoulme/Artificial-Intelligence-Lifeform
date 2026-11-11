"""Probe a candidate's temporary working checkpoint across fresh processes."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

from scripts.deepseek_local_baseline import sha

SOURCE = ROOT/'runs/deepseek-spora-output-v2-checkpoint-20261111/individual-with-output-v2.pt'
OUTPUT = ROOT/'runs/deepseek-spora-candidate-restart-audit-v2-20261111.json'


def child(stage, directory):
    from scripts import deepseek_spora_runtime_chat as runtime
    from scripts.deepseek_named_binding_probe import KEYS
    from spora.growing import query
    from spora.language_training import frozen_digest
    import torch

    torch.set_num_threads(2)
    directory = Path(directory)
    runtime.CHECKPOINT = SOURCE
    runtime.WORKING_CHECKPOINT = directory/'working.pt'
    chat = runtime.load_chat()
    initial_cursor = chat.state.cursor
    if stage == 'write':
        key = torch.tensor([KEYS[0]], dtype=torch.long, device='cuda')
        zeros = torch.zeros_like(key)
        previous_value = int(query(chat.organism, chat.state, key, zeros, zeros).argmax(-1)[0])
        decision = chat.observe_sourced(KEYS[0], 7, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled write was deferred')
        result = {'initial_cursor': initial_cursor, 'cursor': chat.state.cursor, 'previous_native_value': previous_value,
                  'decision': decision, 'individual_digest': frozen_digest(chat.organism, chat.state),
                  'working_sha256': sha(runtime.WORKING_CHECKPOINT)}
    elif stage == 'read':
        result = {'cursor': chat.state.cursor, 'individual_digest': frozen_digest(chat.organism, chat.state),
                  'working_sha256': sha(runtime.WORKING_CHECKPOINT),
                  'question_result': chat.answer('小林现在喜欢喝什么？'),
                  'previous_question_result': chat.answer('小林改口以前喜欢什么饮料？')}
    else:
        raise ValueError('Invalid stage')
    with (directory/f'{stage}.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_hash = sha(SOURCE)
    expected = json.loads((SOURCE.parent/'result.json').read_text(encoding='utf-8'))['candidate_checkpoint_sha256']
    if source_hash != expected:
        raise ValueError('Candidate source digest mismatch')
    with tempfile.TemporaryDirectory(prefix='spora-candidate-restart-') as directory:
        for stage in ('write', 'read'):
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', stage, directory],
                                       capture_output=True, timeout=240)
            if completed.returncode:
                raise RuntimeError(f'{stage} subprocess failed ({completed.returncode}): '
                                   + completed.stderr.decode('utf-8', errors='replace')[-1500:])
        written = json.loads((Path(directory)/'write.json').read_text(encoding='utf-8'))
        restored = json.loads((Path(directory)/'read.json').read_text(encoding='utf-8'))
    if sha(SOURCE) != source_hash:
        raise ValueError('Candidate bootstrap changed')
    from scripts.deepseek_semantic_value_pilot import WORDS
    previous = written['previous_native_value']
    previous_answer = restored['previous_question_result'].get('answer')
    result = {'status': 'controlled_two_process_working_checkpoint_audit',
              'source_sha256': source_hash, 'write': written, 'read': restored,
              'cursor_preserved': restored['cursor'] == written['cursor'],
              'state_digest_preserved': restored['individual_digest'] == written['individual_digest'],
              'working_file_preserved': restored['working_sha256'] == written['working_sha256'],
              'answer_exact': restored['question_result'].get('answer') in ('酸奶', '酸奶。', '酸奶.'),
              'previous_exact': (previous_answer in (WORDS[previous], WORDS[previous]+'。', WORDS[previous]+'.')
                                 if 0 <= previous < len(WORDS) else None),
              'limitations': ['Two controlled processes; neither is forcibly terminated.',
                              'One manually addressed finite write and one authored question.',
                              'Temporary full checkpoint; no incremental storage or real user state.',
                              'The default runtime is not changed to the candidate.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('cursor_preserved', 'state_digest_preserved',
                                             'working_file_preserved', 'answer_exact', 'previous_exact')}), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--child':
        child(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 1:
        run()
    else:
        raise SystemExit('Invalid arguments')
