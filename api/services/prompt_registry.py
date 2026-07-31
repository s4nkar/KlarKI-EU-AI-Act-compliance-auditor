"""Prompt version registry for LangGraph agent nodes (services/agent_graph.py).

Prompts are plain text files under prompts/agent_graph/<node>/v<N>.txt, with
an active version per node tracked in prompts/agent_graph/registry.json —
the same v1/v2 versioning convention already used for trained models (see
training/version_manager.py), just without the model-artifact machinery
(hashing, symlink promotion) that a text file doesn't need.

Placeholders use a {{TOKEN}} (double-brace) convention and are filled via
plain str.replace(), not str.format() — the prompts themselves contain
literal single-brace JSON examples ({"score": ...}), so str.format() would
try to parse those as format fields and raise. This mirrors the same
reasoning documented in classifier.py's _load_prompt().

Reads are uncached (fresh from disk each call) so flipping the active
version in registry.json takes effect on the next request, with no API
restart — same convention as classifier.py's _load_prompt().

Usage:
    from services.prompt_registry import load_prompt
    template = load_prompt("technical_agent")
    prompt = template.replace("{{REQ_STR}}", req_str).replace("{{USER_TEXT}}", user_text)
"""

import json
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "agent_graph"
_REGISTRY_PATH = _PROMPTS_DIR / "registry.json"


def _load_registry() -> dict:
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_active_version(node: str) -> str:
    """Return the active prompt version string for a node (e.g. 'v2')."""
    section = _load_registry().get(node)
    if not section or not section.get("active"):
        raise KeyError(f"No active prompt version configured for node '{node}' in {_REGISTRY_PATH}")
    return section["active"]


def load_prompt(node: str, version: str | None = None) -> str:
    """Load a node's prompt template text.

    Uses the registry's active version unless a specific version is
    explicitly requested (e.g. to A/B two versions in an eval script).
    """
    ver = version or get_active_version(node)
    path = _PROMPTS_DIR / node / f"{ver}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")
