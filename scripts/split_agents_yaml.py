"""One-shot: split configs/agents.yaml into configs/agents/<id>.yaml files.

Run once, then configs/agents.yaml can be deleted (load_agents now prefers the
configs/agents/ directory when it exists). Requires PyYAML with sort_keys
support (i.e. PyYAML >= 5.1). Run from the project root, in the venv.
"""
from __future__ import annotations

import pathlib
import re

import yaml

SRC = pathlib.Path("configs/agents.yaml")
DST_DIR = pathlib.Path("configs/agents")

# Stable field order — matches how we authored agents.yaml by hand.
ORDER = [
    "id", "display_name", "base_url", "model_id", "api_key_env",
    "max_tokens", "temperature", "reasoning",
    "wait_between_requests_s", "notes",
]


def main() -> None:
    data = yaml.safe_load(SRC.read_text()) or {}
    DST_DIR.mkdir(parents=True, exist_ok=True)

    count = 0
    for a in data.get("agents", []):
        aid = a["id"]
        ordered: dict = {}
        for k in ORDER:
            if k in a:
                ordered[k] = a[k]
        # Preserve any fields we didn't list explicitly (lossless).
        for k, v in a.items():
            if k not in ordered:
                ordered[k] = v

        fname = re.sub(r"[^a-zA-Z0-9._-]", "_", aid) + ".yaml"
        (DST_DIR / fname).write_text(yaml.safe_dump(ordered, sort_keys=False))
        count += 1

    print(f"wrote {count} files to {DST_DIR}/")


if __name__ == "__main__":
    main()
