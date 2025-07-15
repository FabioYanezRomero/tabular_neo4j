from __future__ import annotations
"""Utility helpers for JSON serialization of pipeline states.

This module provides a `json_default` function that can be passed to
`json.dump(..., default=json_default)` so that complex objects used in the
pipeline (e.g. `GraphState`, `MultiTableGraphState`, `pandas.DataFrame`, etc.)
are converted to a JSON-serialisable representation rather than falling back to
`str(obj)` which produces huge and unhelpful strings.

The strategy is:
1. `GraphState` -> serialise to a shallow dict of its mapping items but DROP
   large dataframe objects (`raw_dataframe`, `processed_dataframe`). These can
   be reconstructed from CSVs and unnecessarily bloat JSON files.
2. `pandas.DataFrame` -> return `None` to indicate omission.
3. `set` -> convert to list for JSON compatibility.
4. Anything else -> fall back to `str(obj)`.
"""

from typing import Any, Dict
import pandas as pd

try:
    from Tabular_to_Neo4j.app_state import GraphState
except ImportError:  # during some partial builds GraphState might not exist yet
    GraphState = None  # type: ignore

__all__ = ["json_default"]

def _serialise_graph_state(state: "GraphState") -> Dict[str, Any]:
    """Convert a `GraphState` into a JSON-friendly dict.

    DataFrame objects are omitted because they can be huge; all other fields are
    included. Extra dynamic keys stored in `_extra` are also included.
    """
    # Retrieve mapping items (includes _extra via .items()) but exclude DFS
    excluded_keys = {"raw_dataframe", "processed_dataframe"}
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if k in excluded_keys:
            continue
        out[k] = v
    return out

def json_default(obj: Any):  # noqa: ANN401 – signature required by json
    """Default handler for `json.dump`/`json.dumps`.

    Example
    -------
    >>> json.dumps(data, default=json_default)
    """
    if GraphState is not None and isinstance(obj, GraphState):
        return _serialise_graph_state(obj)
    if isinstance(obj, pd.DataFrame):
        return None  # omit heavy DataFrames
    if isinstance(obj, set):
        return list(obj)
    # Add more custom handlers if needed.
    return str(obj)
