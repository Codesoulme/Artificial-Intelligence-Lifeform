"""Observable repetition indicators, not semantic quality judgments."""
from collections import Counter
import re


def repetition_metrics(text):
    grams = [text[i:i + 4] for i in range(max(0, len(text) - 3))]
    direct = bool(re.search(r'(.{2,20})\1{3,}', text))
    lines = [re.sub(r'^\s*(?:\d+[.、．:：)）]|[-*])\s*', '', s).strip()
             for s in text.splitlines()]
    counts = Counter(s for s in lines if len(s) >= 2)
    repeated_lines = max(counts.values(), default=0)
    return dict(repeated_4gram_fraction=1-len(set(grams))/len(grams) if grams else 0.,
                direct_loop=direct, max_repeated_line_count=repeated_lines,
                obvious_loop=direct or repeated_lines >= 4)
