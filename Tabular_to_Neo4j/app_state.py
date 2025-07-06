from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Iterator
from collections.abc import MutableMapping
import pandas as pd


@dataclass
class GraphState(MutableMapping):
    def __init__(self, *args, **kwargs):
        # Remove _extra from kwargs if present
        extra = kwargs.pop('_extra', None)
        # Call dataclass-generated __init__
        super().__init__()
        # Manually assign dataclass fields
        for field_name, field_def in self.__dataclass_fields__.items():
            if field_name in kwargs:
                setattr(self, field_name, kwargs.pop(field_name))
            else:
                setattr(self, field_name, field_def.default_factory() if callable(field_def.default_factory) else field_def.default)
        # Anything left in kwargs is dynamic
        self._extra = kwargs.get('_extra', {}) if extra is None else extra
        if not isinstance(self._extra, dict):
            self._extra = {}
    """State container used throughout the analysis pipeline."""

    csv_file_path: str = ""
    raw_dataframe: Optional[pd.DataFrame] = None
    has_header_heuristic: Optional[bool] = None
    header_row_if_present: Optional[List[str]] = None

    # Header processing state
    inferred_header: Optional[List[str]] = None
    validated_header: Optional[List[str]] = None
    is_header_correct_llm: Optional[bool] = None
    header_correction_suggestions: Optional[str] = None
    translated_header: Optional[List[str]] = None
    is_header_in_target_language: Optional[bool] = None
    final_header: Optional[List[str]] = None
    processed_dataframe: Optional[pd.DataFrame] = None

    # Column analysis state
    column_analytics: Optional[Dict[str, Dict[str, Any]]] = None
    llm_column_semantics: Optional[Dict[str, Dict[str, Any]]] = None

    # Schema synthesis intermediate states
    entity_property_classification: Optional[Dict[str, Dict[str, Any]]] = None
    entity_property_consensus: Optional[Dict[str, Dict[str, Any]]] = None
    entity_relationships: Optional[List[Dict[str, Any]]] = None
    property_entity_mapping: Optional[Dict[str, str]] = None
    cypher_query_templates: Optional[List[Dict[str, Any]]] = None

    # Final output
    inferred_neo4j_schema: Optional[Dict[str, Any]] = None
    error_messages: List[str] = field(default_factory=list)

    # Container for dynamic extra fields
    _extra: Dict[str, Any] = field(default_factory=dict, repr=False, init=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphState":
        """Create a `[GraphState](cci:2://file:///app/Tabular_to_Neo4j/app_state.py:8:0-86:52)` instance from a dictionary."""
        # Remove _extra from data before passing to constructor
        data = dict(data)  # make a copy
        extra = data.pop('_extra', {})
        instance = cls()
        for key, value in data.items():
            instance[key] = value
        # Restore any extra fields
        if isinstance(extra, dict):
            instance._extra.update(extra)
        return instance

    # Mapping protocol methods -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        return self._extra[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.__dataclass_fields__:
            setattr(self, key, value)
        else:
            self._extra[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self.__dataclass_fields__:
            setattr(self, key, None)
        else:
            del self._extra[key]

    def __iter__(self) -> Iterator[str]:
        for key in self.__dataclass_fields__:
            yield key
        for key in self._extra:
            yield key

    def __len__(self) -> int:
        return len(self.__dataclass_fields__) + len(self._extra)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self.__dataclass_fields__ or key in self._extra

    def get(self, key: str, default: Any = None) -> Any:
        return self[key] if key in self else default

    def copy(self) -> "GraphState":
        import copy
        new_state = GraphState.from_dict(dict(self.items()))
        # Deep copy _extra explicitly
        new_state._extra = copy.deepcopy(self._extra)
        return new_state


class MultiTableGraphState(dict):
    """
    State container for multi-table workflows.
    Each key is a table name, and the value is a GraphState for that table.
    """
    def __getitem__(self, key: str) -> GraphState:
        return super().__getitem__(key)

    def __setitem__(self, key: str, value) -> None:
        from collections.abc import MutableMapping
        # Attempt to import `AddableValuesDict` from LangChain. If it is not available,
        # fall back to `None`. We explicitly *cast* the imported symbol to a
        # ``type[Any] | None`` so that static type checkers recognise it as a valid
        # ``class_or_tuple`` argument for ``isinstance``.
        from typing import Any, Type, cast

        try:
            from langchain_core.utils import AddableValuesDict as _AddableValuesDict  # type: ignore
        except ImportError:
            _AddableValuesDict = None  # pragma: no cover

        AddableValuesDict = cast("Type[Any] | None", _AddableValuesDict)
        # Only call isinstance with AddableValuesDict when the import succeeded.
        if not (
            isinstance(value, (GraphState, MutableMapping)) or
            (AddableValuesDict is not None and isinstance(value, AddableValuesDict))
        ):
            raise ValueError(
                f"Value for key '{key}' must be a GraphState, AddableValuesDict, or MutableMapping instance. Got {type(value)}"
            )
        super().__setitem__(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)
