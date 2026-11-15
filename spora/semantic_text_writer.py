"""Learned text-to-semantic-vector writing for phrase slots."""
import torch
from torch import nn


class TextSemanticWriter(nn.Module):
    """Map frozen decoder token states into persistent semantic vectors."""

    def __init__(self, width):
        super().__init__()
        if type(width) is not int or width < 2:
            raise ValueError('Writer width must be a positive hidden width')
        self.projection = nn.Linear(width, width, bias=False, dtype=torch.float32)
        nn.init.normal_(self.projection.weight, std=width ** -0.5)

    def forward(self, hidden):
        if not isinstance(hidden, torch.Tensor) or hidden.shape[-1] != self.projection.in_features:
            raise ValueError('Writer input has an incompatible hidden width')
        return self.projection(hidden.float())


def value_positions(tokenizer, fact, value):
    """Return tokenizer positions overlapping the value span in a fact."""
    encoded = tokenizer(fact, return_tensors='pt', add_special_tokens=False,
                        return_offsets_mapping=True)
    offsets = encoded.pop('offset_mapping')[0].tolist()
    start = fact.index(value)
    end = start + len(value)
    positions = [index for index, (left, right) in enumerate(offsets)
                 if left < end and right > start]
    token_ids = encoded['input_ids'][0].tolist()
    if not positions or [token_ids[index] for index in positions] != tokenizer.encode(value, add_special_tokens=False):
        raise ValueError('Fact value is not aligned to its token span')
    return encoded, positions


def encode_value_sequence(model, tokenizer, writer, fact, value):
    """Encode a fact value from frozen hidden states and append the EOS vector."""
    encoded, positions = value_positions(tokenizer, fact, value)
    device = next(model.parameters()).device
    inputs = {key: tensor.to(device) for key, tensor in encoded.items()}
    with torch.inference_mode():
        outputs = model.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden = outputs.hidden_states[-1][0].index_select(
        0, torch.tensor(positions, device=device)).float()
    predicted = writer(hidden)
    eos = model.model.embed_tokens.weight[tokenizer.eos_token_id].detach().float().unsqueeze(0)
    return torch.cat((predicted, eos), dim=0).detach()
