"""Compare query-dependent memory-slot attention with the replay control."""
import json
from scripts import deepseek_spora_teacher_diverse_memory_pilot as configuration

pilot = configuration.pilot
pilot.OUT = pilot.p.ROOT / 'runs/continuous-reader-cross-attention-pilot-20261113'
pilot.RECOVERY_FILE = pilot.p.ROOT / 'runs/continuous-reader-cross-attention-pilot-20261113-recovery.pt'
pilot.TEACHER_KL_WEIGHT = 0.0
pilot.CROSS_ATTENTION_MEMORY = True
pilot.CROSS_ATTENTION_LAYER = 23
pilot.CROSS_ATTENTION_WIDTH = 64


if __name__ == '__main__':
    result = json.loads((pilot.START_CANDIDATE.parent / 'result.json').read_text(encoding='utf-8'))
    if pilot.p.sha(pilot.START_CANDIDATE) != result['candidate_sha256']:
        raise ValueError('Starting candidate checksum mismatch')
    pilot.main()
