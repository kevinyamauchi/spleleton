"""Config dataclasses to and from JSON, and their canonical hash.

Every config of the library is a (nested) dataclass of plain values. Here
they are turned into plain Python types (:func:`to_dict`), rebuilt from them
(:func:`from_dict`), written as canonical JSON (sorted keys, tuples as lists,
no whitespace) and hashed (:func:`spec_hash`). Two configs that compare equal
have the same hash, whatever their field order or the process that made them.
"""

import dataclasses
import hashlib
import json
import types
import typing

import numpy as np


def to_dict(config: typing.Any) -> dict:
    """Convert a config dataclass to plain Python types (e.g. for JSON)."""
    return _plain(dataclasses.asdict(config))


def from_dict(cls: type, data: dict) -> typing.Any:
    """Rebuild a (nested) config dataclass from plain Python types.

    Fields missing from ``data`` keep their defaults, so a config saved before
    a field was added still loads. Keys that are not fields are an error.

    Parameters
    ----------
    cls : type
        The dataclass to build.
    data : dict
        The output of :func:`to_dict` (or its JSON round trip).

    Returns
    -------
    object
        An instance of ``cls``.
    """
    hints = typing.get_type_hints(cls)
    fields = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"{cls.__name__} has no fields {sorted(unknown)}")
    return cls(**{name: _convert(hints[name], value) for name, value in data.items()})


def canonical_json(config: typing.Any) -> str:
    """Canonical JSON of a config: sorted keys, tuples as lists, no spaces."""
    data = to_dict(config) if dataclasses.is_dataclass(config) else _plain(config)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def spec_hash(config: typing.Any) -> str:
    """SHA-256 (hex) of a config's :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def _plain(value: typing.Any) -> typing.Any:
    """Plain JSON types, recursively.

    Dataclasses become dicts, tuples lists, and NumPy scalars Python numbers.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _convert(annotation: typing.Any, value: typing.Any) -> typing.Any:
    if value is None:
        return None
    origin = typing.get_origin(annotation)
    if isinstance(annotation, types.UnionType) or origin is typing.Union:
        options = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        return _convert(_pick(options, value), value)
    if dataclasses.is_dataclass(annotation):
        return from_dict(annotation, value)
    if origin is tuple:
        args = typing.get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], item) for item in value)
        if args:
            return tuple(
                _convert(arg, item) for arg, item in zip(args, value, strict=True)
            )
        return tuple(value)
    return value


def _pick(options: list[typing.Any], value: typing.Any) -> typing.Any:
    """The option of a union that a value belongs to.

    A dict goes to the first dataclass that has all of its keys as fields;
    anything else goes to the first option.
    """
    if isinstance(value, dict):
        for option in options:
            if dataclasses.is_dataclass(option):
                names = {f.name for f in dataclasses.fields(option)}
                if set(value) <= names:
                    return option
    return options[0]
