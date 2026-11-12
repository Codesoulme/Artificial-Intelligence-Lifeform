"""Train correction-aware memory output with multi-entity and question variation."""
from scripts import deepseek_spora_semantic_bridge_pilot as pilot


pilot.OUT = pilot.p.ROOT / 'runs/continuous-reader-multi-entity-question-finetune-pilot-20261112'
pilot.CODEC_SLOT_COUNT = 5
pilot.CODEC_ADD_EOS = True
pilot.STORE_FULL_FACT = True
pilot.STORE_IDENTITY = True
pilot.BRIDGE_CONDITION_SLOTS = 4
pilot.TRAIN_VALUE_COUNT = len(pilot.WORDS)
pilot.TRAIN_CORRECTION_SEQUENCES = True
pilot.TRAIN_MULTI_ENTITY_STATES = True
pilot.TRAIN_EXTRA_QUESTIONS = True
pilot.DIRECT_RESIDUAL = True
pilot.DIRECT_RESIDUAL_LAYER = 20
pilot.STRICT_SWAP_EVAL = True
pilot.START_CANDIDATE = pilot.p.ROOT / 'runs/continuous-reader-correction-sequence-pilot-20261112/candidate.pt'
pilot.LANGUAGE_LR = 2e-5
pilot.main()
