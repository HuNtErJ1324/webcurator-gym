"""A small fixed offline fallback corpus.

Used only when the real tokenized validation set cannot be loaded or decoded.
It is intentionally generic and self-contained so leakage scoring still works
offline, with explicit fallback provenance and a warning.
"""

from __future__ import annotations

DEFAULT_EVAL_CORPUS: list[str] = [
    "The mitochondrion is the powerhouse of the cell, generating most of the "
    "chemical energy needed to power the cell's biochemical reactions.",
    "In computer science, a binary search algorithm finds the position of a "
    "target value within a sorted array by repeatedly halving the search interval.",
    "Photosynthesis is the process by which green plants convert sunlight, water, "
    "and carbon dioxide into glucose and oxygen.",
    "The French Revolution was a period of radical political and societal change "
    "in France that began in 1789 and ended in the late 1790s.",
    "A neural network is a series of algorithms that endeavors to recognize "
    "underlying relationships in a set of data through a process that mimics the "
    "way the human brain operates.",
    "Supply and demand is an economic model of price determination in a market: "
    "the price of a good adjusts until the quantity demanded equals the quantity "
    "supplied.",
    "The speed of light in a vacuum is approximately 299,792 kilometers per "
    "second and is a universal physical constant denoted by the letter c.",
    "Shakespeare's play Hamlet tells the story of a Danish prince who seeks "
    "revenge against his uncle for murdering his father and seizing the throne.",
]
