"""
Shared prompt loader for agent prompts.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tomllib


_PROMPTS_PATH = Path(__file__).with_name("prompts.toml")


@lru_cache(maxsize=1)
def _load_prompts() -> dict:
    with _PROMPTS_PATH.open("rb") as fh:
        return tomllib.load(fh)


def get_prompt(section: str, key: str = "system") -> str:
    prompts = _load_prompts()
    return str(prompts[section][key])
