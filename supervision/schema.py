"""Tiny JSON-Schema subset validator (no dependencies).

Tool arguments come from a model, so they are untrusted input: validate them
exactly like you would validate a public API request.
"""
from __future__ import annotations

import re

_PY = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}


def validate(schema: dict, value, path: str = "args") -> list:
    errs: list = []
    t = schema.get("type")
    if t:
        ok = isinstance(value, _PY[t])
        if t in ("integer", "number") and isinstance(value, bool):
            ok = False
        if not ok:
            return [f"{path}: expected {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: must be one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: must be <= {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            errs.append(f"{path}: does not match {schema['pattern']}")
    if t == "object":
        props = schema.get("properties", {})
        for r in schema.get("required", []):
            if r not in value:
                errs.append(f"{path}.{r}: required")
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    errs.append(f"{path}.{k}: unexpected field")
        for k, sub in props.items():
            if k in value:
                errs += validate(sub, value[k], f"{path}.{k}")
    if t == "array" and "items" in schema:
        for i, v in enumerate(value):
            errs += validate(schema["items"], v, f"{path}[{i}]")
    return errs
