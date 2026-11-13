"""Opt-in research helper requiring an explicitly supplied name per read.

User input is not authentication or proof of entity identity. This does not
resolve homonyms, aliases, or an incorrect confirmation by the user.
"""
from .identity_read_guard import IdentityReadGuard


class ConfirmedNameGuard(IdentityReadGuard):
    def __init__(self, fingerprints, tokenizer, question, name):
        super().__init__(fingerprints, tokenizer)
        if not isinstance(question, str) or not question.strip():
            raise ValueError('A nonempty question is required')
        if not isinstance(name, str) or not name:
            raise ValueError('An explicitly supplied name is required')
        self.question = question
        self.name = name

    def authorize(self, question, selected, key):
        return (question == self.question and self.selected_name(question, selected) == self.name
                and super().authorize(question, selected, key))


def answer_with_confirmed_name(chat, question, name=None):
    """Ask for a name before routing; never persist the user's confirmation."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError('A nonempty question is required')
    if name is None:
        return {'accepted': False, 'answer': None, 'reason': 'explicit_name_required'}
    guard = ConfirmedNameGuard(chat.organism.identity_fingerprints, chat.tokenizer, question, name)
    try:
        key = guard.fingerprints.lookup(name)
    except ValueError:
        key = None
    if key is None:
        return {'accepted': False, 'answer': None, 'reason': 'explicit_name_not_enrolled'}
    return chat.answer(question, identity_guard=guard)
