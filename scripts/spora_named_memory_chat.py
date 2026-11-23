"""Local Chinese dialogue and explicitly addressed persistent memory."""

import argparse
from pathlib import Path

from scripts.audit_named_memory_interleaved import choose_modes
from scripts.audit_qwen_compatible_natural_memory_20261121 import MODEL, WORDS, extract, load_head, sha, torch
from scripts.audit_qwen_durable_entity_reads_20261122 import read_value
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_durable import create_integrated_durable, load_integrated_durable
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
DEFAULT_SOURCE = RUNS / 'qwen-compatible-memory-only-layer9-v1' / 'integrated.pt'
DEFAULT_PACKED = RUNS / 'named-memory-layer9-packed-reply-v1' / 'packed.pt'
DEFAULT_TIME = RUNS / 'qwen-memory-only-layer9-temporal-v1' / 'factorized'
DEFAULT_RECOVERY = RUNS / 'named-memory-time-expanded-v1'
DEFAULT_DATABASE = RUNS / 'named-memory-local' / 'individual.sqlite'


class LocalChat:
    def __init__(self, runtime, tokenizer, temporal, recovery):
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.temporal = temporal
        self.recovery = recovery
        self.turns = []

    def register(self, name):
        result = self.runtime.resolve_confirmed_name(name)
        if result['key'] is not None:
            return '这个姓名已经登记。'
        if result['reason'] == 'invalid_confirmed_name':
            return '请提供有效的完整姓名。'
        model = self.runtime.organism
        bindings = getattr(model, 'identity_fingerprints', None)
        if bindings is None:
            if self.runtime.state.cursor:
                return '现有个体的事实地址尚未绑定姓名。'
            key = 0
        else:
            free = (~bindings.occupied).nonzero(as_tuple=True)[0]
            if not len(free):
                return '姓名登记已满。'
            key = int(free[0])
        self.runtime.enroll_name(key, name)
        return f'已登记{name}。'

    def remember(self, name, value):
        address = self.runtime.resolve_confirmed_name(name)['key']
        if address is None:
            return '请先登记这个完整姓名。'
        if value not in WORDS:
            return '目前可记的饮品有' + '、'.join(WORDS) + '。'
        decision, _ = self.runtime.observe_sourced(address, WORDS.index(value), 'direct', 1.)
        if decision['current_write']:
            return '已记录。'
        if decision['action'] == 'confirm':
            return '这项记录原本就是这个值。'
        return '本次没有更新当前值。'

    def recall(self, name, question):
        address = self.runtime.resolve_confirmed_name(name)['key']
        if address is None:
            return '没有登记这个完整姓名。'
        hidden, mask = extract(self.runtime.decoder.base, self.tokenizer, [question], batch_size=1,
                               hidden_indices=tuple(self.temporal['hidden_indices']), last_logit_only=True)
        mode = choose_modes(self.temporal, hidden, mask, [question])[0]
        if mode == 3 and choose_modes(self.recovery, hidden, mask, [question])[0] == 2:
            mode = 2
        if mode == 3:
            return '请说明要查的是当前值、更正前的值，还是更正后的值。'
        if self.runtime.state.cursor == 0:
            return '还没有可读的饮品记录。'
        result = read_value(self.runtime, self.tokenizer, question, address, mode)
        if result['text'] and result['text'].strip('。！! ') == result['value']:
            return result['text']
        return '这次没能可靠读出饮品记录。'

    @torch.inference_mode()
    def talk(self, text):
        messages = [*self.turns, {'role': 'user', 'content': text}]
        while True:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt + '</think>\n\n', return_tensors='pt', add_special_tokens=False)
            length = inputs['input_ids'].shape[1]
            if length <= 1024:
                break
            if len(messages) == 1:
                return '这条消息太长，请缩短一点。'
            messages = messages[2:]
        output = self.runtime.generate_ordinary(**inputs.to('cuda'), max_new_tokens=96, do_sample=False,
                                                use_cache=True, temperature=None, top_p=None,
                                                eos_token_id=self.tokenizer.eos_token_id,
                                                pad_token_id=self.tokenizer.pad_token_id)
        reply = self.tokenizer.decode(output[0, length:], skip_special_tokens=True).strip()
        if reply:
            self.turns = [*messages, {'role': 'assistant', 'content': reply}][-8:]
        return reply or '这次没有生成回复。'

    def handle(self, line):
        if not line.startswith('/'):
            return self.talk(line)
        command, *args = line.split(maxsplit=2)
        if command == '/登记' and len(args) == 1:
            return self.register(args[0])
        if command == '/记' and len(args) == 2:
            return self.remember(*args)
        if command == '/查' and len(args) == 2:
            return self.recall(*args)
        return '用 /登记 姓名、/记 姓名 饮品、/查 姓名 问题，或输入 /退出。'


def main(args):
    torch.set_num_threads(2)
    source_digest = sha(args.source)
    temporal, _ = load_head(args.temporal, 'temporal.pt', 'temporal_sha256', source_digest)
    recovery, _ = load_head(args.recovery, 'temporal.pt', 'temporal_sha256', source_digest)
    from scripts.audit_qwen_packed_reply_20261122 import compare_payloads

    compare_payloads(args.source, args.checkpoint)
    if temporal['hidden_indices'] != recovery['hidden_indices']:
        raise ValueError('Time selectors require different decoder layers')
    args.database.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    if args.database.exists():
        runtime = load_integrated_durable(args.database, args.checkpoint, base, MODEL)
    else:
        runtime = create_integrated_durable(args.database, args.checkpoint, base, MODEL,
                                            initial_state_factory=fresh_controlled_state)
    with runtime:
        print('孢子已就绪。可直接聊天，也可用 /登记、/记、/查。输入 /退出结束。', flush=True)
        chat = LocalChat(runtime, tokenizer, temporal, recovery)
        while True:
            try:
                line = input('你 > ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line == '/退出':
                break
            if line:
                try:
                    print('孢子 >', chat.handle(line), flush=True)
                except ValueError as error:
                    print('孢子 >', str(error), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (('source', DEFAULT_SOURCE), ('checkpoint', DEFAULT_PACKED), ('temporal', DEFAULT_TIME),
                          ('recovery', DEFAULT_RECOVERY), ('database', DEFAULT_DATABASE)):
        parser.add_argument('--' + name, type=Path, default=default)
    main(parser.parse_args())
