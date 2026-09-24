"""Shared token and quoted-string handling for OSLO prescription data."""

from __future__ import annotations

import re


def tokenize(statement: str) -> list[str]:
    """Split whitespace/comma-separated tokens, preserving quoted strings."""
    return re.findall(r'"(?:\\.|[^"\\])*"|[^\s,]+', statement)


def decode_text(value: str) -> str:
    """Remove one pair of delimiters and decode escaped quotes/backslashes."""
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return re.sub(r'\\(["\\])', r"\1", value)
