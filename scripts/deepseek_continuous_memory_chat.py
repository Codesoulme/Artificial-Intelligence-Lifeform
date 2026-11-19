"""Interactive read-only chat for the continuous-memory research checkpoint."""

import argparse
import json
import sys

from deepseek_aligned_memory_calibration import load_adapter
from deepseek_aligned_memory_dialogue import OUT as ADAPTER_RUN, RANK, generate_with_memory
from deepseek_memory_attribute_router import CLARIFY, MISSING
from deepseek_memory_attribute_span import Route, route_action
from deepseek_memory_cosine_readout import cosine_bias, projected_codebook
from deepseek_memory_dialogue_context import OUT as DIALOGUE_RUN, features_for, route
from deepseek_memory_field_decoding import FIELDS, OUT as FIELD_RUN
from deepseek_memory_field_transfer import OUT as TRANSFER_RUN
from deepseek_memory_turn_priority import OUT as PRIORITY_RUN, gate_probabilities
from deepseek_output_aligned_memory import MODEL, SEED, SOURCE, fresh, load_integrated, read, sha, torch

ENTITIES = (26, 47)


class ContinuousMemoryChat:
    def __init__(self, decoder, tokenizer, site, banks, styles, timing, columns, contextual, direct, inheritance, setup):
        self.decoder, self.tokenizer, self.site, self.banks = decoder, tokenizer, site, banks
        self.styles, self.timing, self.columns = styles, timing, columns
        self.contextual, self.direct, self.inheritance = contextual, direct, inheritance
        self.gate_cutoff = setup['gate_cutoff']
        self.cutoff = setup['cutoffs']['priority']
        weight = decoder.base.lm_head.weight
        basis = torch.linalg.qr(torch.randn(weight.shape[1], RANK, generator=torch.Generator().manual_seed(SEED))).Q
        self.codes, self.scale = projected_codebook(weight, basis.to(weight.device))

    @torch.inference_mode()
    def answer(self, question, entity, version, history=None):
        if entity not in ENTITIES or version not in (0, 1):
            raise ValueError('Unknown experimental address or snapshot')
        row = dict(key='input', question=question, history=history, target_turn=0)
        features = features_for(self.decoder, self.tokenizer, row, False)
        probability, = gate_probabilities(self.inheritance, [row], {'input': features})
        inherits = bool(history) and probability >= self.gate_cutoff
        router = self.contextual if inherits else self.direct
        if inherits:
            features = features_for(self.decoder, self.tokenizer, row, True)
        predicted, = route(router, [row], {'input': features})
        selected = predicted['selected']
        action = route_action(predicted, self.cutoff)
        start = None
        reason = action
        if action == 'read':
            address = self.site.current_basis[entity:entity + 1]
            bias = cosine_bias(self.site, self.banks[selected, version], address, self.codes, self.scale)
            if bias is None:
                action, reason, response = 'missing', 'empty_matrix', MISSING
            else:
                tokens, start = generate_with_memory(self.decoder, self.tokenizer, features[3], self.styles[selected],
                                                      self.columns, self.timing, 2 * bias)
                if start is None:
                    action, reason, response = 'clarify', 'injection_not_triggered', CLARIFY
                else:
                    response = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        elif action == 'ordinary':
            ids = torch.tensor([features[3]], device=self.codes.device)
            tokens = self.decoder.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                           max_new_tokens=64, pad_token_id=self.tokenizer.pad_token_id,
                                           eos_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(tokens[0, ids.shape[1]:], skip_special_tokens=True).strip()
        else:
            response = MISSING if action == 'missing' else CLARIFY
        return dict(response=response, action=action, reason=reason, field=selected, confidence=predicted['confidence'],
                    inherits=inherits, memory_start=start, entity=entity, version=version)


def load_chat():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    paths = dict(adapter=ADAPTER_RUN / 'candidate.pt', styles=FIELD_RUN / 'field-adapters.pt',
                 context=DIALOGUE_RUN / 'router.pt', priority=PRIORITY_RUN / 'routing.pt',
                 routes=PRIORITY_RUN / 'routes.json', banks=TRANSFER_RUN / 'banks.pt')
    expected = {paths['adapter']: read(ADAPTER_RUN / 'result.json')['candidate_sha256']}
    for name in ('styles', 'context', 'priority', 'routes', 'banks'):
        path = paths[name]
        expected[path] = read(path.parent / 'result.json')['artifact_sha256'][path.name]
    model_files = (SOURCE, *(MODEL / name for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json')))
    plan_path = TRANSFER_RUN / 'plan.json'
    if sha(plan_path) != read(TRANSFER_RUN / 'result.json')['artifact_sha256'][plan_path.name]:
        raise ValueError(f'Artifact digest mismatch: {plan_path}')
    provenance = read(plan_path)['protected']
    expected.update({path: provenance[str(path)] for path in model_files})
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError(f'Artifact digest mismatch: {path}')
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
              dtype=torch.float16, attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, organism, _ = load_integrated(SOURCE, base, MODEL, individual_device='cpu')
    decoder.eval().requires_grad_(False)
    width = base.lm_head.weight.shape[1]
    _, _, timing, columns = load_adapter(torch.load(paths['adapter'], weights_only=True), width)
    checkpoint = torch.load(paths['styles'], weights_only=True)
    styles = {}
    for field in FIELDS:
        style = torch.nn.Sequential(torch.nn.Linear(width, 24), torch.nn.Tanh(), torch.nn.Linear(24, len(columns)))
        style.load_state_dict(checkpoint[field]['style'])
        styles[field] = style.eval().requires_grad_(False).to('cuda')
    contextual = Route(width, localized=True).eval()
    contextual.load_state_dict(torch.load(paths['context'], weights_only=True))
    priority = torch.load(paths['priority'], weights_only=True)
    direct = Route(width, localized=True).eval()
    direct.load_state_dict(priority['direct'])
    inheritance = torch.nn.Linear(width, 1).eval()
    inheritance.load_state_dict(priority['inheritance'])
    banks = torch.load(paths['banks'], weights_only=True)
    return ContinuousMemoryChat(decoder, tokenizer, organism.memories[0], banks, styles, timing, columns,
                                contextual, direct, inheritance, read(paths['routes']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entity', type=int, choices=ENTITIES, required=True)
    parser.add_argument('--version', type=int, choices=(0, 1), default=1)
    parser.add_argument('--question')
    parser.add_argument('--history', help='Preceding user turn')
    parser.add_argument('--json', action='store_true', help='Include routing and read diagnostics')
    args = parser.parse_args()
    chat = load_chat()
    history = args.history
    lines = sys.stdin if args.question is None else (args.question,)
    if args.question is None and sys.stdin.isatty():
        print('Enter one question per line. Ctrl+Z then Enter exits on Windows.', file=sys.stderr, flush=True)
    for line in lines:
        question = line.strip()
        if not question:
            continue
        result = chat.answer(question, args.entity, args.version, history)
        print(json.dumps(result, ensure_ascii=False) if args.json else result['response'], flush=True)
        history = question


if __name__ == '__main__':
    main()
