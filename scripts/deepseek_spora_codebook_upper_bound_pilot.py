"""Run the semantic pilot with a fixed internal codebook upper bound."""
from scripts import deepseek_spora_semantic_bridge_pilot as pilot


class NearestTokenDecoder(pilot.p.torch.nn.Module):
    """Decode compressed slots with the frozen base vocabulary codebook.

    This is a diagnostic upper bound for the learned bridge.  It is not
    presented as open memory because it performs an explicit finite-codebook
    lookup.
    """

    def __init__(self, codec, embedding):
        super().__init__()
        self.register_buffer('codebook', codec.codebook.detach().to(dtype=pilot.p.torch.float16).clone())
        self.register_buffer('embedding', pilot.p.torch.nn.functional.normalize(
            embedding.detach().to(dtype=pilot.p.torch.float16), dim=-1))
        self._cache_key = None
        self._cache_value = None

    def set_cache_key(self, key):
        if key != self._cache_key:
            self._cache_value = None
        self._cache_key = key

    def forward(self, codes):
        if self._cache_key is not None and self._cache_value is not None:
            return self._cache_value.to(device=codes.device)
        flat = pilot.p.torch.nn.functional.normalize(codes.float().reshape(-1, codes.shape[-1]), dim=-1).to(
            dtype=self.codebook.dtype)
        scores = flat @ self.codebook.T
        values, indices = scores.topk(4, dim=-1)
        del scores
        weights = (values / 0.03).softmax(-1).to(dtype=self.embedding.dtype)
        decoded = (weights.unsqueeze(-1) * self.embedding[indices]).sum(-2).float()
        decoded = pilot.p.torch.nn.functional.normalize(decoded, dim=-1).reshape_as(
            codes.new_empty(codes.shape[0], codes.shape[1], self.embedding.shape[1]))
        self._cache_value = decoded.detach()
        return decoded.to(device=codes.device)


def make_token_decoder(codec, embedding):
    return NearestTokenDecoder(codec, embedding).to(embedding.device).eval()


def fit_token_decoder(decoder, codec, embedding, **kwargs):
    return [{'step': 0, 'cosine_loss': 0.0,
             'method': 'frozen_codebook_top4_temperature_0.03'}]


def run():
    pilot.make_token_decoder = make_token_decoder
    pilot.fit_token_decoder = fit_token_decoder
    pilot.OUT = pilot.p.ROOT / 'runs/continuous-reader-codebook-upper-bound-pilot-20261112'
    pilot.main()


if __name__ == '__main__':
    run()
