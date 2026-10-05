"""Safe extraction of common Obsidian Markdown metadata."""

import json
import re
from typing import Any

import yaml

FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
WIKILINK = re.compile(r"!?\[\[([^\]]+)\]\]")
TAG = re.compile(r"(?<![#\w/])#([A-Za-z0-9_/-]+)")


def parse_obsidian_markdown(text: str) -> tuple[dict[str, Any], list[str]]:
    """Return frontmatter/link/tag metadata and non-fatal parsing warnings."""

    warnings: list[str] = []
    metadata: dict[str, Any] = {}
    match = FRONTMATTER.match(text)
    body = text
    if text.startswith("---"):
        if match is None:
            warnings.append("Malformed YAML frontmatter: closing '---' delimiter not found")
        else:
            body = text[match.end() :]
            try:
                loaded = yaml.safe_load(match.group(1))
            except yaml.YAMLError as error:
                warnings.append(f"Malformed YAML frontmatter: {error}")
            else:
                if loaded is None:
                    loaded = {}
                if isinstance(loaded, dict):
                    metadata = json.loads(json.dumps(loaded, default=str))
                else:
                    warnings.append("YAML frontmatter must contain a mapping")

    aliases = _as_strings(metadata.get("aliases"))
    tags = _as_strings(metadata.get("tags"))
    inline_tags = TAG.findall(body)
    links: list[dict[str, str | bool]] = []
    for match in WIKILINK.finditer(body):
        item = match.group(1)
        target, separator, alias = item.partition("|")
        target = target.split("#", maxsplit=1)[0].strip()
        if target:
            links.append(
                {
                    "target": target,
                    "alias": alias.strip() if separator else "",
                    "embedded": match.group(0).startswith("!"),
                }
            )
    metadata.update(
        {
            "aliases": list(dict.fromkeys(aliases)),
            "tags": list(dict.fromkeys(tag.lstrip("#") for tag in [*tags, *inline_tags])),
            "outgoing_links": links,
        }
    )
    return metadata, warnings


def remove_frontmatter(text: str) -> str:
    """Remove only a valid leading frontmatter block from extracted note text."""

    match = FRONTMATTER.match(text)
    return text[match.end() :].strip() if match else text.strip()


def _as_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if isinstance(item, (str, int, float))]
    return []
