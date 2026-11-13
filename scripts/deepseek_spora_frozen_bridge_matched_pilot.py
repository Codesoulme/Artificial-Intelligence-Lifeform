"""Matched control: continue the same candidate with frozen semantic bridge."""
import json
from scripts import deepseek_spora_semantic_bridge_pilot as pilot

pilot.OUT = pilot.p.ROOT / 'runs/continuous-reader-frozen-bridge-matched-pilot-20261113'
pilot.RECOVERY_FILE = pilot.p.ROOT / 'runs/continuous-reader-frozen-bridge-matched-pilot-20261113-recovery.pt'
pilot.CODEC_SLOT_COUNT = 5
pilot.CODEC_ADD_EOS = True
pilot.STORE_FULL_FACT = True
pilot.STORE_IDENTITY = True
pilot.BRIDGE_CONDITION_SLOTS = 4
pilot.WORDS += ('山竹', '石榴', '杨梅', '龙眼', '桑葚', '枇杷', '莲雾', '蛇果', '百香果', '椰子')
pilot.TRAIN_VALUE_COUNT = 22
pilot.TRAIN_CORRECTION_SEQUENCES = True
pilot.TRAIN_MULTI_ENTITY_STATES = True
pilot.TRAIN_EXTRA_QUESTIONS = False
pilot.DIRECT_RESIDUAL = True
pilot.DIRECT_RESIDUAL_LAYER = 20
pilot.STRICT_SWAP_EVAL = True
pilot.START_CANDIDATE = pilot.p.ROOT / 'runs/continuous-reader-balanced-multifruit-pilot-20261113/candidate.pt'
pilot.REPLAY_VALUE_START = 18
pilot.REPLAY_VALUE_FACTOR = 4
pilot.EVALUATION_VALUES = (0, 1, 3, 9, 16, 17, *range(18, 28))
pilot.EVALUATE_FIRST_MULTI_QUESTION_ONLY = True
pilot.TRAIN_BRIDGE = False
pilot.BRIDGE_LANGUAGE_LR = 2e-6
pilot.LANGUAGE_LR = 2e-5
pilot.LANGUAGE_STEPS = 768


if __name__ == '__main__':
    result = json.loads((pilot.START_CANDIDATE.parent / 'result.json').read_text(encoding='utf-8'))
    if pilot.p.sha(pilot.START_CANDIDATE) != result['candidate_sha256']:
        raise ValueError('Starting candidate checksum mismatch')
    pilot.main()
