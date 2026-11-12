"""Test semantic memory when the complete fact token sequence is stored."""
from scripts import deepseek_spora_semantic_bridge_pilot as pilot


pilot.OUT = pilot.p.ROOT / 'runs/continuous-reader-full-fact-semantic-bridge-pilot-20261112'
pilot.CODEC_SLOT_COUNT = 5
pilot.CODEC_ADD_EOS = True
pilot.STORE_FULL_FACT = True
pilot.BRIDGE_CONDITION_SLOTS = 4
pilot.main()
