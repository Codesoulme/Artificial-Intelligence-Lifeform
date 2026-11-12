"""Text reconstruction with a nonlinear continuous-memory semantic bridge."""
from scripts import deepseek_spora_text_reconstruction_pilot as pilot


pilot.OUT = pilot.p.ROOT / 'runs/continuous-reader-text-reconstruction-mlp-pilot-20261113'
pilot.BRIDGE_KIND = 'mlp'
pilot.BRIDGE_STEPS = 800
pilot.DECODER_STEPS = 1200
pilot.main()
