"""Remove only explicit leading question/answer labels from reviewed QA."""
import re

QUESTION = re.compile(r'^\s*(?:请回答以下问题|问题|问)\s*[:：]\s*')
ANSWER = re.compile(r'^\s*(?:答案|回答|答)\s*[:：]\s*')


def strip_qa_labels(question, answer):
    if not isinstance(question, str) or not isinstance(answer, str):
        raise ValueError('Question and answer must be strings')
    question, answer = question.strip(), answer.strip()
    # A dataset record can have both an instruction label and a question label.
    for _ in range(2):
        question = QUESTION.sub('', question, count=1)
    answer = ANSWER.sub('', answer, count=1)
    if not question or not answer:
        raise ValueError('Label removal left an empty message')
    return question, answer
