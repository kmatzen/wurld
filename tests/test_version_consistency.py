"""The version is declared in three places; they must agree where they overlap.

``pyproject.toml`` states the version statically rather than reading
``wurld.__version__``, so the two can drift silently — a release would then ship
a wheel whose metadata disagrees with ``wurld.__version__`` at runtime, and
nothing would catch it until someone compared them by hand.

``package.json`` is deliberately *not* required to match: the npm package
(``wurld-core``) and the PyPI package are at different points in their history,
which ``.github/workflows/publish-npm.yml`` explains. What is checked here is
that it carries a valid version at all, since a malformed one fails only at
publish time, after the tag exists.
"""

import re
import tomllib
import json
from pathlib import Path

import wurld

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^\d+\.\d+\.\d+([-+].*)?$")


def test_pyproject_matches_the_package():
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert declared == wurld.__version__, (
        f"pyproject.toml says {declared}, wurld.__version__ says {wurld.__version__}; "
        "both are written by hand and must be bumped together")


def test_every_declared_version_is_wellformed():
    npm = json.loads((ROOT / "package.json").read_text())["version"]
    assert SEMVER.match(npm), f"package.json version {npm!r} is not semver"
    assert SEMVER.match(wurld.__version__), f"{wurld.__version__!r} is not semver"


def test_the_changelog_has_a_section_for_this_version():
    """A release tag freezes the tree, so an unstamped section cannot be fixed after."""
    text = (ROOT / "CHANGELOG.md").read_text()
    heading = re.search(
        rf"^## {re.escape(wurld.__version__)}(?: |$)(.*)$", text, re.M)
    assert heading, f"CHANGELOG.md has no '## {wurld.__version__}' section"
    assert "unreleased" not in heading.group(0).lower(), (
        f"the {wurld.__version__} section is still marked unreleased; stamp it with "
        "the date before tagging")
