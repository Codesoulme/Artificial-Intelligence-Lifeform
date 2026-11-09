"""Construct page-scoped Wikipedia context candidates for review only.

Inputs must come from the verified P230 retained candidate and its pinned source.
This module does not open source files, select pages, train, or inspect test data.
Offsets are Python Unicode code points, with exclusive ends.
"""

import hashlib
import re

from scripts import prepare_p230_wikipedia_body as p230
from scripts import refine_p230_wikipedia_body as refined


STATUS = 'pending_review_not_approved_for_training'
LIST = re.compile(r'^(?:[*#•●▪]|\d+[.、)]|[一二三四五六七八九十]+、)')
QUOTE = re.compile(r'^(?:[>「『“]|(?:引用|引文)\s*[:：])')
HEADING = re.compile(r'^(?:={2,}\s*(.+?)\s*={2,}|([^。！？!?；;：:]{1,30}))$')


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def classify_line(text):
    """Classify conservatively; damaged and unsafe material is never a heading."""
    if refined.reasons(text) or p230.NOISE.search(text):
        return 'damaged'
    if LIST.match(text):
        return 'list'
    if QUOTE.match(text):
        return 'quote'
    if HEADING.fullmatch(text):
        return 'heading'
    if p230.rejection(text):
        return 'damaged'
    return 'paragraph'


def _safe_label(text):
    return (isinstance(text, str) and bool(text.strip()) and len(text) <= 200
            and '\n' not in text and '\r' not in text
            and not p230.NOISE.search(text) and not refined.reasons(text))


def _validated_spans(source, passages):
    """Validate every retained passage against the same source page."""
    text = source['text']
    spans = set()
    for row in passages:
        if (row.get('status') != STATUS or row.get('page_id') != source['id']
                or row.get('title') != source['title'] or row.get('url') != source['url']
                or row.get('source_text_sha256') != digest(text)):
            raise ValueError('Passage source mismatch')
        start, end = row['char_start'], row['char_end']
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or row.get('text') != text[start:end] or row.get('text_sha256') != digest(text[start:end])):
            raise ValueError('Passage text or offset mismatch')
        para = row.get('paragraphs')
        if not isinstance(para, list) or not para:
            raise ValueError('Missing paragraph spans')
        previous = start
        for entry in para:
            a, b = entry['char_start'], entry['char_end']
            if (type(a) is not int or type(b) is not int or not previous <= a < b <= end
                    or text[previous:a].strip() or (a, b) in spans):
                raise ValueError('Invalid paragraph spans')
            spans.add((a, b))
            previous = b
        if para[0]['char_start'] != start or previous != end:
            raise ValueError('Incomplete paragraph spans')
    return spans


def build_page_candidates(source, passages, count_tokens):
    """Build candidates from one allowed page; never add unretained body text.

    count_tokens accepts text and returns its BPE count without special tokens.
    A heading is context only, and is cleared at every non-retained line.
    """
    if (not isinstance(source.get('id'), str) or not source['id'].isdecimal()
            or not isinstance(source.get('text'), str) or not _safe_label(source.get('title'))):
        raise ValueError('Invalid source page')
    spans = _validated_spans(source, passages)
    lines = list(p230.paragraphs(source['text']))
    actual = {(a, b) for a, b, _ in lines}
    if not spans <= actual:
        raise ValueError('Paragraph is not an original line')
    results, current, section = [], [], None

    def flush():
        if not current:
            return
        start, end = current[0][0], current[-1][1]
        body = source['text'][start:end]
        labels = [source['title']]
        if section and section['text'] != source['title']:
            labels.append(section['text'])
        prefix = '\n'.join(labels) + '\n\n'
        visible = prefix + body
        body_tokens, label_tokens, visible_tokens = map(count_tokens, (body, prefix, visible))
        if any(type(n) is not int or n < 0 for n in (body_tokens, label_tokens, visible_tokens)):
            raise ValueError('Invalid token count')
        results.append({'status': STATUS, 'page_id': source['id'], 'title': source['title'],
                        'url': source['url'], 'source_text_sha256': digest(source['text']),
                        'section': section, 'char_start': start, 'char_end': end,
                        'paragraphs': [{'char_start': a, 'char_end': b, 'kind': 'paragraph'}
                                       for a, b in current],
                        'text': body, 'text_sha256': digest(body), 'model_text': visible,
                        'body_bpe_tokens': body_tokens, 'extra_heading_bpe_tokens': label_tokens,
                        'model_bpe_tokens': visible_tokens,
                        'model_minus_body_bpe_tokens': visible_tokens - body_tokens})
        current.clear()

    for start, end, line in lines:
        if p230.END_SECTION.fullmatch(line):
            flush()
            break
        kind = classify_line(line)
        if (start, end) in spans and kind == 'paragraph':
            current.append((start, end))
            continue
        flush()
        section = None
        if kind == 'heading' and _safe_label(line):
            section = {'text': line, 'char_start': start, 'char_end': end}
    flush()
    return results


def sample_candidates(candidates, reviewed_rounds, count, seed):
    """Select one candidate per page, excluding both complete review rounds."""
    if (len(reviewed_rounds) != 2 or any(len(r) != 40 for r in reviewed_rounds)
            or type(count) is not int or count < 1 or not isinstance(seed, str) or not seed):
        raise ValueError('Expected two 40-page review rounds and a valid sample request')
    rounds = [set(r) for r in reviewed_rounds]
    if (any(len(r) != 40 or any(not isinstance(p, str) or not p.isdecimal() for p in r) for r in rounds)
            or rounds[0] & rounds[1]):
        raise ValueError('Review pages must be unique across rounds')
    excluded = rounds[0] | rounds[1]
    pages = {}
    for row in candidates:
        pid = row['page_id']
        if row.get('status') != STATUS:
            raise ValueError('Invalid candidate status')
        if pid in excluded:
            continue
        rank = digest('span\0' + seed + '\0' + pid + '\0' + str(row['char_start']))
        if pid not in pages or rank < pages[pid][0]:
            pages[pid] = (rank, row)
    if len(pages) < count:
        raise ValueError('Insufficient eligible pages')
    selected = sorted(pages, key=lambda pid: digest('page\0' + seed + '\0' + pid))[:count]
    return [pages[pid][1] for pid in selected]
