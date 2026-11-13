"""Continue from the pre-multi-entity reader with corrected multi-memory labels."""
from scripts import deepseek_spora_semantic_bridge_pilot as pilot

pilot.OUT = pilot.p.ROOT / 'runs/continuous-reader-corrected-multifruit-pilot-20261113'
pilot.CODEC_SLOT_COUNT = 5
pilot.CODEC_ADD_EOS = True
pilot.STORE_FULL_FACT = True
pilot.STORE_IDENTITY = True
pilot.BRIDGE_CONDITION_SLOTS = 4
pilot.WORDS += ('山竹', '石榴', '杨梅', '龙眼', '桑葚', '枇杷', '莲雾', '蛇果', '百香果', '椰子')
pilot.TRAIN_VALUE_COUNT = 22  # Original 18 plus four; final six are held out.
pilot.TRAIN_CORRECTION_SEQUENCES = True
pilot.TRAIN_MULTI_ENTITY_STATES = True
pilot.TRAIN_EXTRA_QUESTIONS = False
pilot.DIRECT_RESIDUAL = True
pilot.DIRECT_RESIDUAL_LAYER = 20
pilot.STRICT_SWAP_EVAL = True
pilot.START_CANDIDATE = pilot.p.ROOT / 'runs/continuous-reader-correction-sequence-pilot-20261112/candidate.pt'
pilot.RECOVERY_FILE = pilot.p.ROOT / 'runs/continuous-reader-corrected-multifruit-pilot-20261113-recovery.pt'
pilot.LANGUAGE_LR = 2e-5
pilot.LANGUAGE_STEPS = 512


if __name__ == '__main__':
    pilot.main()
