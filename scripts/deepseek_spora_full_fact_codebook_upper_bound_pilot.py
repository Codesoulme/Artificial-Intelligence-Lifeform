"""Diagnostic upper bound for complete fact-token storage."""
from scripts import deepseek_spora_codebook_upper_bound_pilot as upper


upper.pilot.OUT = upper.pilot.p.ROOT / 'runs/continuous-reader-full-fact-codebook-upper-bound-pilot-20261112'
upper.pilot.CODEC_SLOT_COUNT = 5
upper.pilot.CODEC_ADD_EOS = True
upper.pilot.STORE_FULL_FACT = True
upper.pilot.BRIDGE_CONDITION_SLOTS = 4
upper.pilot.make_token_decoder = upper.make_token_decoder
upper.pilot.fit_token_decoder = upper.fit_token_decoder
upper.pilot.main()
