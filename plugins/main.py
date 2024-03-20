from __future__ import annotations as _annotations

import logging
import os
import re
from pathlib import Path

from mkdocs.config import Config
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

logger = logging.getLogger("mkdocs.plugin")
THIS_DIR = Path(__file__).parent
DOCS_DIR = THIS_DIR.parent
PROJECT_ROOT = DOCS_DIR.parent


def on_pre_build(config: Config) -> None:
    """
    Before the build starts.
    """
    add_changelog()


def on_files(files: Files, config: Config) -> Files:
    """
    After the files are loaded, but before they are read.
    """
    return files


def on_page_markdown(markdown: str, page: Page, config: Config, files: Files) -> str:
    """
    Called on each file after it is read and before it is converted to HTML.
    """
    if md := add_version(markdown, page):
        return md
    else:
        return markdown


def add_changelog() -> None:
    history = (PROJECT_ROOT / "HISTORY.md").read_text()
    history = re.sub(
        r"#(\d+)", r"[#\1](https://github.com/pydantic/pydantic/issues/\1)", history
    )
    history = re.sub(
        r"(\s)@([\w\-]+)", r"\1[@\2](https://github.com/\2)", history, flags=re.I
    )
    history = re.sub("@@", "@", history)
    new_file = DOCS_DIR / "changelog.md"

    # avoid writing file unless the content has changed to avoid infinite build loop
    if not new_file.is_file() or new_file.read_text() != history:
        new_file.write_text(history)


MIN_MINOR_VERSION = 7
MAX_MINOR_VERSION = 11


def add_version(markdown: str, page: Page) -> str | None:
    if page.file.src_uri != "index.md":
        return None

    version_ref = os.getenv("GITHUB_REF")
    if version_ref and version_ref.startswith("refs/tags/"):
        version = re.sub("^refs/tags/", "", version_ref.lower())
        url = f"https://github.com/pydantic/pydantic/releases/tag/{version}"
        version_str = f"Documentation for version: [{version}]({url})"
    elif sha := os.getenv("GITHUB_SHA"):
        url = f"https://github.com/pydantic/pydantic/commit/{sha}"
        sha = sha[:7]
        version_str = f"Documentation for development version: [{sha}]({url})"
    else:
        version_str = "Documentation for development version"
    markdown = re.sub(r"{{ *version *}}", version_str, markdown)
    return markdown
