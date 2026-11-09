"""Build review-only contiguous spans from one structured Wikibooks page.

Offsets refer to Unicode code points in the original wikitext, with exclusive
ends. Spans do not establish that their contents are self-contained or suitable
for training; surrounding page context still requires review.
"""

from scripts.wikibooks_structured import extract


def build_spans(raw, extraction):
    """Combine consecutive review candidates, preserving their source gaps.

    Pass the original page and the result of ``wikibooks_structured.extract``.
    Every quarantined paragraph is a hard boundary, including headings. The
    original paragraphs and their link metadata are retained in each span.
    """
    if not isinstance(raw, str):
        raise TypeError('raw must be a string')
    if not isinstance(extraction, dict) or extraction.get('offset_unit') != 'unicode_codepoint':
        raise ValueError('expected a structured extraction with Unicode offsets')
    if extraction != extract(raw):
        raise ValueError('extraction does not match source')
    paragraphs = extraction.get('paragraphs')
    if not isinstance(paragraphs, list):
        raise ValueError('expected a paragraph list')

    spans = []
    current = []
    previous_end = 0

    def flush():
        if not current:
            return
        start, end = current[0]['start'], current[-1]['end']
        text = current[0]['text']
        for before, after in zip(current, current[1:]):
            text += raw[before['end']:after['start']] + after['text']
        spans.append({'start': start, 'end': end, 'raw': raw[start:end],
                      'text': text, 'paragraphs': current[:]})
        current.clear()

    for paragraph in paragraphs:
        if not isinstance(paragraph, dict):
            raise ValueError('invalid paragraph')
        start, end = paragraph.get('start'), paragraph.get('end')
        if (type(start) is not int or type(end) is not int or
                start < previous_end or end <= start or end > len(raw) or
                paragraph.get('raw') != raw[start:end] or
                raw[previous_end:start].strip()):
            raise ValueError('invalid paragraph offsets or source gap')
        status = paragraph.get('status')
        if status == 'review_candidate':
            if not isinstance(paragraph.get('text'), str) or paragraph.get('risks') != []:
                raise ValueError('invalid review candidate')
            current.append(paragraph)
        elif status == 'quarantined':
            flush()
        else:
            raise ValueError('unknown paragraph status')
        previous_end = end
    if raw[previous_end:].strip():
        raise ValueError('unaccounted source text')
    flush()
    return {'status': 'review_only_not_approved_for_training',
            'offset_unit': 'unicode_codepoint', 'spans': spans,
            'limitations': ['Spans may depend on omitted page context and are not necessarily self-contained.',
                            'Factual accuracy, attribution, rights and overlap require review.']}
