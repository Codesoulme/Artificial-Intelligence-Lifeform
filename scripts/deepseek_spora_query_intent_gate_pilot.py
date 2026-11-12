"""Train a question-conditioned gate for memory residual injection."""
from scripts import deepseek_spora_semantic_bridge_pilot as pilot


pilot.OUT = pilot.p.ROOT / 'runs/continuous-reader-query-intent-gate-pilot-20261113'
pilot.CODEC_SLOT_COUNT = 5
pilot.CODEC_ADD_EOS = True
pilot.STORE_FULL_FACT = True
pilot.STORE_IDENTITY = True
pilot.BRIDGE_CONDITION_SLOTS = 4
pilot.TRAIN_VALUE_COUNT = len(pilot.WORDS)
pilot.TRAIN_CORRECTION_SEQUENCES = False
pilot.TRAIN_MULTI_ENTITY_STATES = False
pilot.TRAIN_EXTRA_QUESTIONS = True
pilot.FOCUS_EXTRA_QUESTIONS = True
pilot.EXTRA_QUESTION_TEMPLATES = (
    '请查阅记忆后告诉我{name}爱吃什么水果，只回答水果名称。',
    '关于{name}爱吃的水果，请回答名称。',
    '记忆里的{name}水果偏好是什么？只写名称。',
)
pilot.DIRECT_RESIDUAL = True
pilot.DIRECT_RESIDUAL_LAYER = 20
pilot.QUERY_INTENT_GATE = True
pilot.STRICT_SWAP_EVAL = True
pilot.START_CANDIDATE = pilot.p.ROOT / 'runs/continuous-reader-multi-entity-question-finetune-pilot-20261112/candidate.pt'
pilot.LANGUAGE_LR = 1e-5
pilot.LANGUAGE_STEPS = 256
pilot.main()
