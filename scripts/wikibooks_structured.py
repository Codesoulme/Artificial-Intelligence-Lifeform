"""Conservative, review-only extraction of Wikibooks wikitext.

Offsets are zero-based Python character offsets into the original string; end
offsets are exclusive. A paragraph containing any uncertain markup is withheld
in full, since removing even a small fragment can change the surrounding prose.
"""
import re

import mwparserfromhell
from mwparserfromhell.nodes import Text, Wikilink


_PARAGRAPH_BREAK = re.compile(r'\r?\n[\t \r]*\n(?:[\t \r]*\n)*')
_UNPARSED_MARKUP = re.compile(
    r'\{\{|\}\}|\[\[|\]\]|\{\||\|\}|-\{|\}-|'
    r'<|>|&(?:\#\w+|\w+);|https?://|www\.|'
    r'(?m:^[ \t]*[\*#;:!|=])', re.I)
_IMAGE = re.compile(r'^\s*:?(?:file|image|文件|檔案|档案|图像|圖像|图片|圖片)\s*:', re.I)


def _risk(reason, start, end, raw):
    return {'reason': reason, 'start': start, 'end': end, 'raw': raw[start:end]}


def _link(node, start, end, raw):
    target = str(node.title)
    label = str(node.text) if node.text is not None else target
    if _IMAGE.match(target):
        return None, _risk('image', start, end, raw)
    # Namespace, interwiki, pipe-trick, and nested markup have context-dependent
    # rendering or effects. Only literal text with an explicit target is kept.
    if (not target.strip() or ':' in target or not label.strip()
            or target != target.strip() or '\n' in target or '\r' in target
            or '\n' in label or '\r' in label
            or any(not isinstance(part, Text) for part in node.title.nodes)
            or (node.text is not None and any(not isinstance(part, Text) for part in node.text.nodes))
            or _UNPARSED_MARKUP.search(target) or _UNPARSED_MARKUP.search(label)
            or '|' in target or '|' in label):
        return None, _risk('ambiguous_link', start, end, raw)
    return {'start': start, 'end': end, 'raw': raw[start:end],
            'target': target, 'text': label}, None


def _node_reason(node):
    kind = type(node).__name__
    if kind == 'Template' or kind == 'Argument':
        return 'template'
    if kind == 'ExternalLink':
        return 'external_link'
    if kind == 'Heading':
        return 'heading'
    if kind == 'Comment':
        return 'comment'
    if kind == 'HTMLEntity':
        return 'entity'
    if kind == 'Tag':
        tag = str(node.tag).strip().lower()
        if tag in ('ref', 'references', 'blockquote', 'cite', 'q'):
            return 'citation'
        if tag in ('math', 'chem'):
            return 'formula'
        if tag in ('code', 'pre', 'source', 'syntaxhighlight'):
            return 'code'
        if tag in ('gallery', 'imagemap'):
            return 'image'
        if tag in ('table', 'tr', 'td', 'th') or tag == '{|':
            return 'table'
        return 'tag_or_formatting'
    return 'unsupported_' + kind.lower()


def extract(raw):
    """Return paragraph candidates and quarantined material for human review.

    Candidates are not independently complete, fact-checked, rights-cleared, or
    approved for training. Inspect the whole page and its source context.
    """
    if not isinstance(raw, str):
        raise TypeError('raw must be a string')
    nodes = mwparserfromhell.parse(raw).nodes
    paragraphs = []
    start = 0
    offset = 0
    for node in nodes:
        value = str(node)
        if isinstance(node, Text):
            for match in _PARAGRAPH_BREAK.finditer(value):
                end = offset + match.start()
                if raw[start:end].strip():
                    paragraphs.append((start, end))
                start = offset + match.end()
        offset += len(value)
    if raw[start:].strip():
        paragraphs.append((start, len(raw)))

    results = []
    uncertain_boundary = False
    for start, end in paragraphs:
        segment = raw[start:end]
        risks, links, rendered = [], [], []
        if uncertain_boundary:
            risks.append(_risk('uncertain_prior_markup_boundary', start, end, raw))
        offset = start
        for node in mwparserfromhell.parse(segment).nodes:
            value = str(node)
            stop = offset + len(value)
            if isinstance(node, Text):
                rendered.append(value)
                for match in _UNPARSED_MARKUP.finditer(value):
                    risks.append(_risk('unparsed_markup', offset + match.start(),
                                       offset + match.end(), raw))
            elif isinstance(node, Wikilink):
                link, risk = _link(node, offset, stop, raw)
                if risk:
                    risks.append(risk)
                else:
                    links.append(link)
                    rendered.append(link['text'])
            else:
                risks.append(_risk(_node_reason(node), offset, stop, raw))
            offset = stop
        if any(risk['reason'] == 'unparsed_markup' for risk in risks):
            uncertain_boundary = True
        results.append({'start': start, 'end': end, 'raw': segment,
                        'text': None if risks else ''.join(rendered),
                        'links': links, 'risks': risks,
                        'status': 'quarantined' if risks else 'review_candidate'})
    return {'status': 'review_only_not_approved_for_training',
            'offset_unit': 'unicode_codepoint', 'paragraphs': results,
            'limitations': ['Context, factual accuracy, attribution, rights and overlap require review.',
                            'Markup detection is conservative and may not cover every MediaWiki extension.']}
