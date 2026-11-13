"""Same expanded data, start and steps as the teacher pilot, without teacher KL."""
import json
from scripts import deepseek_spora_teacher_diverse_memory_pilot as configuration

pilot = configuration.pilot
pilot.OUT = pilot.p.ROOT / 'runs/continuous-reader-teacher-diverse-control-20261113'
pilot.RECOVERY_FILE = pilot.p.ROOT / 'runs/continuous-reader-teacher-diverse-control-20261113-recovery.pt'
pilot.TEACHER_KL_WEIGHT = 0.0


if __name__ == '__main__':
    result = json.loads((pilot.START_CANDIDATE.parent / 'result.json').read_text(encoding='utf-8'))
    if pilot.p.sha(pilot.START_CANDIDATE) != result['candidate_sha256']:
        raise ValueError('Starting candidate checksum mismatch')
    pilot.main()
