"""Smoke-test candidate delegation with a temporary runtime checkpoint."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import deepseek_spora_runtime_chat as runtime
from spora.observation_candidate import ObservationCandidate


def main():
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / 'individual.pt'
        runtime.WORKING_CHECKPOINT = destination
        chat = runtime.load_chat()
        initial_cursor = chat.state.cursor
        candidate = ObservationCandidate(6, '咖啡', 1, 'affirmative', 'direct', 1.0,
                                         initial_cursor, .99, '小林改成喜欢咖啡')
        decision = chat.observe_candidate(candidate)
        if not destination.is_file():
            raise AssertionError('Candidate delegation did not create a temporary checkpoint')
        fresh_base = runtime.AutoModelForCausalLM.from_pretrained(
            runtime.MODEL, local_files_only=True, trust_remote_code=False,
            torch_dtype=runtime.torch.float16, attn_implementation='sdpa').to('cuda').eval()
        fresh_base.requires_grad_(False)
        reloaded_decoder, reloaded_organism, reloaded_state = runtime.load_integrated(
            destination, fresh_base, runtime.MODEL, individual_device='cuda')
        del reloaded_decoder, reloaded_organism
        result = {'status': 'isolated_candidate_persistence_smoke', 'decision': decision,
                  'initial_cursor': initial_cursor, 'saved_cursor': chat.state.cursor,
                  'reloaded_cursor': reloaded_state.cursor,
                  'working_sha256': runtime.sha(destination),
                  'real_working_checkpoint_modified': False}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
