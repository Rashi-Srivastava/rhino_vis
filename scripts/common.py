from __future__ import annotations
import json, logging, re
from pathlib import Path
from typing import Any
import yaml

def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def parse_observation_filename(path: str | Path, regex: str) -> dict[str, str]:
    path = Path(path)
    match = re.match(regex, path.name)
    if match is None:
        raise ValueError(f"Unexpected filename: {path.name}")
    date = match.group("date")
    time_token = match.group("time")
    return {"date": date, "time": time_token, "display_name": f"{date} {time_token.replace('-', ':')}"}

def ensure_observation_dirs(config, obs):
    plots_day = Path(config["paths"]["plots_root"]) / obs["date"]
    outputs_day = Path(config["paths"]["outputs_root"]) / obs["date"]
    dirs = {
        "outputs_day": outputs_day,
        "raw": plots_day / "raw",
        "rfi": plots_day / "rfi",
        "cleaned": plots_day / "cleaned",
        "noise": plots_day / "noise",
        "states": plots_day / "states",
        "diagnostics": plots_day / "diagnostics",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs

def configure_logging(path=None):
    handlers = [logging.StreamHandler()]
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=handlers, force=True)

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
