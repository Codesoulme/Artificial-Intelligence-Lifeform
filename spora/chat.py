"""Read-only local console for the P18 language interface and symbol memory.

The default mode keeps requests independent. Optional multi-turn mode keeps a
bounded in-process transcript only; it never writes the transcript to neural
memory or disk.
"""
import argparse
import hashlib
import sys
import unicodedata
from pathlib import Path
import torch
from .growing import load, query
from .language_bridge import LanguageBridge
from .language_training import load_interface
from .language_eval import generate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT/'checkpoints/individual-prechinese-v3.pt'
DEFAULT_INTERFACE = ROOT/'runs/chinese-curriculum-1000/checkpoint.pt'


def visible(text):
    """Keep generated text readable without executing terminal control bytes."""
    return ''.join(c if c in '\n\t' or not unicodedata.category(c).startswith('C')
                   else '\\u'+format(ord(c), '04x') for c in text.replace('\r\n', '\n'))


def bounded_conversation_prompt(history, current, budget, max_turns=8):
    """Build a bounded prompt from RAM-only turns, keeping the newest turns."""
    if type(budget) is not int or budget < 1 or type(max_turns) is not int or max_turns < 1:
        raise ValueError('Invalid conversation budget')
    turns = list(history)[-max_turns:]
    while True:
        parts = [f'用户：{user}\n助手：{assistant}' for user, assistant in turns]
        parts.append(f'用户：{current}\n助手：')
        prompt = '\n'.join(parts)
        if len(prompt.encode('utf-8')) <= budget or not turns:
            return prompt
        turns.pop(0)


class LocalConversation:
    def __init__(self, parent, interface, device='cpu', max_new_bytes=96, multi_turn=False):
        if type(max_new_bytes) is not int or not 1 <= max_new_bytes <= 128:
            raise ValueError('生成长度必须在1到128之间。')
        parent, interface = Path(parent), Path(interface)
        for path in (parent, interface):
            if not path.is_file():
                raise FileNotFoundError('找不到本地模型文件：'+str(path))
        digest = hashlib.sha256(parent.read_bytes()).hexdigest()
        organism, self.state, _ = load(parent, device)
        payload = torch.load(interface, map_location='cpu', weights_only=True)
        self.model = LanguageBridge(organism, context_layers=payload.get('context_layers', 0)).to(device).eval()
        load_interface(self.model, payload, digest)
        self.total_steps = payload.get('steps', 0)+(payload.get('lineage') or {}).get('prior_steps', 0)
        self.max_new_bytes = max_new_bytes
        self.multi_turn = bool(multi_turn)
        self.history = []
        self.input_budget = organism.config.max_length-1-max_new_bytes
        if self.input_budget < 1:
            raise ValueError('生成预算超过当前模型的可用长度。')
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def help(self):
        print('输入短句后按回车，例如：天亮了。 或 国家层面：')
        print(f'每次最多输入{self.input_budget}个UTF-8字节，约{self.input_budget//3}个常用汉字。')
        print('/memory 12     查看键12的当前记忆')
        print('/memory 12 0   查看键12在第0次观察时的记录')
        if self.multi_turn:
            print('/clear         清除本次运行的临时多轮上下文')
        print('/info          查看当前版本；/help 帮助；/quit 退出')

    @torch.no_grad()
    def respond(self, text):
        stripped = text.strip()
        if not stripped:
            return True
        parts = stripped.split()
        if stripped in ('/quit', '/exit'):
            return False
        if stripped == '/clear' and self.multi_turn:
            self.history.clear()
            print('已清除本次运行的临时多轮上下文。')
            return True
        if stripped == '/help':
            self.help()
        elif stripped == '/info':
            print(f'累计中文训练{self.total_steps}步；已保存{self.state.cursor}次符号观察。')
            mode = '启用临时多轮上下文' if self.multi_turn else '每次输入独立续写'
            print(f'{mode}；交流不会自动写入长期记忆。')
        elif parts[0] == '/memory':
            if len(parts) not in (2, 3):
                raise ValueError('用法：/memory 键 [观察时刻]，例如 /memory 12 0')
            try:
                key = int(parts[1])
                event = int(parts[2]) if len(parts) == 3 else None
            except ValueError:
                raise ValueError('键和观察时刻请填写整数。') from None
            if not 0 <= key < self.model.organism.config.key_count:
                raise ValueError(f'键的范围为0到{self.model.organism.config.key_count-1}。')
            if event is not None and not 0 <= event < self.state.cursor:
                raise ValueError(f'已有观察时刻范围为0到{self.state.cursor-1}。')
            device = next(self.model.parameters()).device
            answer = int(query(self.model.organism, self.state,
                torch.tensor([key], device=device), torch.tensor([int(event is not None)], device=device),
                torch.tensor([event if event is not None else 0], device=device)).argmax(-1))
            value = '未知' if answer == self.model.organism.config.unknown else str(answer)
            when = '当前' if event is None else f'第{event}次观察'
            print(f'记忆查询：键{key}，{when}的值为{value}。')
        elif stripped.startswith('/'):
            raise ValueError('未知命令，输入 /help 查看帮助。')
        else:
            if len(text.encode('utf-8')) > self.input_budget:
                raise ValueError(f'这段输入太长，请缩短到约{self.input_budget//3}个常用汉字以内；没有截断或保存它。')
            prompt = bounded_conversation_prompt(self.history, text, self.input_budget) if self.multi_turn else text
            result = generate(self.model, self.state, prompt, self.max_new_bytes)
            output = visible(result['text']).strip()
            print('模型：'+(output if output else '（没有生成可显示的文字）'))
            if not result['utf8_valid']:
                print('提示：原始输出包含无效编码，显示中的替代符号不算有效文字。')
            if self.multi_turn:
                self.history.append((text, output))
                del self.history[:-8]
        return True


def main():
    parser = argparse.ArgumentParser(description='SPORA 本地只读试聊')
    parser.add_argument('--individual', default=str(DEFAULT_PARENT))
    parser.add_argument('--interface', default=str(DEFAULT_INTERFACE))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--max-new-bytes', type=int, default=96)
    parser.add_argument('--multi-turn', action='store_true',
                        help='保留本次进程内的有限对话历史；不写入长期记忆或磁盘')
    parser.add_argument('--prompt', help='只运行一次输入；省略则进入交互窗口')
    args = parser.parse_args()
    # UTF-8 is needed both in Windows terminals and redirected streams.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    torch.set_num_threads(2)
    try:
        print('正在加载本地 SPORA…', flush=True)
        conversation = LocalConversation(args.individual, args.interface, device, args.max_new_bytes, args.multi_turn)
        print(f'已加载累计{conversation.total_steps}步的语言版本。当前只能尝试短文本续写，回答可能不正确。')
        print(('本次启用有限多轮上下文；' if args.multi_turn else '每次输入独立处理；') + '本轮交流不会自动记为长期经历。')
        if args.prompt is not None:
            conversation.respond(args.prompt)
            return
        conversation.help()
        while True:
            try:
                if not conversation.respond(input('\n你：')):
                    break
            except ValueError as error:
                print(str(error))
    except (KeyboardInterrupt, EOFError):
        print('\n已退出。')
    except (OSError, ValueError, RuntimeError) as error:
        print('无法完成本次试聊：'+str(error), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
