"""
Shared rig configuration — eye cam vs world cam assignment.

All services (receiver, engine, calibration_server) read from this.
The TUI writes to it and notifies running services.
"""
import json
import pathlib

_PATH = pathlib.Path(__file__).parent / "rig_config.json"
_DEFAULTS = {"eye_cam": 2, "world_cam": 1}


def load() -> dict:
    try:
        with open(_PATH) as f:
            d = json.load(f)
        return {**_DEFAULTS, **d}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(_DEFAULTS)


def save(cfg: dict):
    current = load()
    current.update(cfg)
    with open(_PATH, "w") as f:
        json.dump(current, f, indent=2)


def eye_cam() -> int:
    return load()["eye_cam"]


def world_cam() -> int:
    return load()["world_cam"]
