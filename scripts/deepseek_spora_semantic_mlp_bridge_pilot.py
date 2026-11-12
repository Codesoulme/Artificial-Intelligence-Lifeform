"""Run the semantic memory pilot with a nonlinear embedding decompressor."""
import torch

from scripts import deepseek_spora_semantic_bridge_pilot as pilot


class TokenCodeDecoder(pilot.p.torch.nn.Module):
    def __init__(self, slot_width, embedding_width):
        super().__init__()
        self.net = pilot.p.torch.nn.Sequential(
            pilot.p.torch.nn.Linear(slot_width, 512),
            pilot.p.torch.nn.GELU(),
            pilot.p.torch.nn.Linear(512, embedding_width),
        )
        pilot.p.torch.nn.init.xavier_uniform_(self.net[0].weight)
        pilot.p.torch.nn.init.xavier_uniform_(self.net[2].weight)

    def forward(self, codes):
        return pilot.p.torch.nn.functional.normalize(self.net(codes.float()), dim=-1)


pilot.TokenCodeDecoder = TokenCodeDecoder
pilot.OUT = pilot.p.ROOT / 'runs/continuous-reader-semantic-mlp-bridge-pilot-20261112'
pilot.BRIDGE_STEPS = 600
pilot.main()
