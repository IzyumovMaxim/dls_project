"""Passage text handling, shared by the offline sentence encoder and the serving path.

Precomputed sentence vectors are addressed by position, so the builder and the app must split
passages identically or the highlight lands on the wrong sentence. Hence one module, not two.
"""

from __future__ import annotations

import re

# FEVER escapes brackets in its Wikipedia dump. Doc ids in our corpus ship decoded; passage text
# still carries -LSB-/-RSB-.
WIKI_TOKENS = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-COLON-": ":",
}

# Passages are tokenized, so a sentence period is its own space-padded token ("in 1996 . After").
# Breaking after any period followed by whitespace covers that and ordinary punctuation, and leaves
# decimals like "3.5" alone.
SENTENCE_BREAK = re.compile(r"(?<=\.)\s+")

MIN_SENTENCE_WORDS = 4


def detokenize(text: str) -> str:
    for token, char in WIKI_TOKENS.items():
        text = text.replace(token, char)
    return text


def split_sentences(text: str) -> list[str]:
    """Every sentence, short ones included, so positions stay stable across builder and app."""
    return [s.strip() for s in SENTENCE_BREAK.split(detokenize(text)) if s.strip()]


def evidence_candidates(sentences: list[str]) -> list[int]:
    """Positions long enough to be scored: fragments score noisily against a claim."""
    return [i for i, s in enumerate(sentences) if len(s.split()) >= MIN_SENTENCE_WORDS]
