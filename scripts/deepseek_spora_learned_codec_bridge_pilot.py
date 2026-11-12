"""Pretrain a learned continuous codec before language conditioning."""
from scripts import deepseek_spora_semantic_bridge_pilot as pilot


CODEC_STEPS = 400


class LearnedCodec(pilot.ContinuousTextCodec):
    """ContinuousTextCodec with a trainable embedding projection."""

    def __init__(self, embedding, **kwargs):
        super().__init__(embedding, **kwargs)
        initial = self.projection.detach().clone()
        del self._buffers['projection']
        self.projection = pilot.p.torch.nn.Parameter(initial)

    @pilot.p.torch.no_grad()
    def refresh_codebook(self, embedding, chunk_size=4096):
        rows = []
        for start in range(0, embedding.shape[0], chunk_size):
            rows.append(pilot.p.torch.nn.functional.normalize(
                embedding[start:start + chunk_size].float() @ self.projection, dim=-1).detach())
        self.codebook.copy_(pilot.p.torch.cat(rows, dim=0))


class LearnedTokenDecoder(pilot.p.torch.nn.Module):
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


def make_token_decoder(codec, embedding):
    torch = pilot.p.torch
    decoder = LearnedTokenDecoder(codec.slot_width, embedding.shape[1]).to(embedding.device)
    optimizer = torch.optim.AdamW([
        {'params': [codec.projection], 'lr': 5e-4},
        {'params': list(decoder.parameters()), 'lr': 2e-3},
    ], weight_decay=1e-4, foreach=False)
    generator = torch.Generator(device=embedding.device).manual_seed(pilot.SEED + 29)
    trace = []
    for step in range(CODEC_STEPS):
        ids = torch.randint(codec.vocab_size, (8192,), generator=generator, device=embedding.device)
        target = torch.nn.functional.normalize(embedding.index_select(0, ids).float(), dim=-1)
        codes = torch.nn.functional.normalize(
            embedding.index_select(0, ids).float() @ codec.projection, dim=-1)
        prediction = decoder(codes)
        loss = 1.0 - (prediction * target).sum(-1).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite learned codec loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([codec.projection, *decoder.parameters()], 1.0,
                                       error_if_nonfinite=True)
        optimizer.step()
        if step in (0, 9, 49, 99, 199, 399):
            trace.append({'step': step + 1, 'cosine_loss': float(loss.detach())})
    codec.refresh_codebook(embedding)
    decoder.slot_count = pilot.CODEC_SLOT_COUNT
    decoder.codec_trace = trace
    return decoder.eval()


def fit_token_decoder(decoder, codec, embedding, **kwargs):
    return getattr(decoder, 'codec_trace', [])


pilot.CODEC_FACTORY = LearnedCodec
pilot.make_token_decoder = make_token_decoder
pilot.fit_token_decoder = fit_token_decoder
pilot.OUT = pilot.p.ROOT / 'runs/continuous-reader-learned-codec-bridge-pilot-20261112'
pilot.CODEC_SLOT_COUNT = 5
pilot.CODEC_ADD_EOS = True
pilot.STORE_FULL_FACT = True
pilot.BRIDGE_CONDITION_SLOTS = 4
pilot.main()
