# SPDX-License-Identifier: Apache-2.0
"""
Tool allowlist filter.

Case-insensitive match on ``tool.name``. Pure function, no external deps.
"""

from typing import TypeVar

T = TypeVar("T")


def filter_tools_by_allowlist(
    tools: list[T],
    allowlist: tuple[str, ...] | list[str] | None,
) -> tuple[list[T], int]:
    """Return tools whose ``name`` is in ``allowlist`` (case-insensitive).

    - ``allowlist=None`` is a no-op: returns ``tools`` unchanged.
    - Empty ``allowlist`` returns an empty list.
    - Returns ``(filtered, removed_count)``.
    """
    if allowlist is None:
        return list(tools), 0

    allowed = {name.lower() for name in allowlist}
    filtered = [
        t for t in tools if getattr(t, "name", "").lower() in allowed
    ]
    return filtered, len(tools) - len(filtered)
