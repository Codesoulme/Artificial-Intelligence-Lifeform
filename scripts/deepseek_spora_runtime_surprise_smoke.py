"""Measure novelty on sequential real-runtime corrections in a temporary copy."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_runtime_chat as runtime
from spora.observation_candidate import ObservationCandidate

OUTPUT = ROOT / 'runs/deepseek-spora-runtime-surprise-smoke-20261111.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / 'individual.pt'
        runtime.WORKING_CHECKPOINT = destination
        chat = runtime.load_chat()
        events = []
        for value, text in ((1, '小林改成喜欢咖啡'), (1, '小林仍然喜欢咖啡'), (6, '小林改回喜欢水')):
            cursor = chat.state.cursor
            candidate = ObservationCandidate(6, '咖啡' if value == 1 else '水', value,
                                             'affirmative', 'direct', 1.0, cursor, .99, text)
            decision = chat.observe_candidate(candidate)
            events.append({'value': value, 'cursor': cursor, 'decision': decision})
        result = {'status': 'isolated_runtime_surprise_smoke', 'events': events,
                  'final_cursor': chat.state.cursor, 'working_sha256': runtime.sha(destination),
                  'real_working_checkpoint_modified': False}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
