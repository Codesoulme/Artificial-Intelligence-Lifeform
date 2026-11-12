"""Test late residual injection with address-bound identity and fact-only pages."""
from scripts import deepseek_spora_semantic_bridge_pilot as pilot


pilot.OUT = pilot.p.ROOT / 'runs/continuous-reader-fact-only-late-residual-strict-pilot-20261112'
pilot.CODEC_SLOT_COUNT = 4
pilot.CODEC_ADD_EOS = True
pilot.STORE_FULL_FACT = True
pilot.STORE_IDENTITY = False
pilot.BRIDGE_CONDITION_SLOTS = 4
pilot.DIRECT_RESIDUAL = True
pilot.DIRECT_RESIDUAL_LAYER = 20
pilot.STRICT_SWAP_EVAL = True
pilot.main()
