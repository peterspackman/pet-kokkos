#!/usr/bin/env python3
"""Package indexes (PEP 503) for pet-kokkos's GPU wheels, from GitHub releases.

The GPU wheels (pet_kokkos-0.1.0+cu12-..., +cu13, +rocm6, +rocm7) are release
assets, not on PyPI. This writes, for each variant, <out>/<variant>/index.html
and <out>/<variant>/pet-kokkos/index.html linking every release's wheels of
that variant, with their sha256, for GitHub Pages; then

    uv pip install pet-kokkos --extra-index-url https://<owner>.github.io/pet-kokkos/cu12

    python tools/wheel_index.py <owner/repo> <out>     (needs gh, logged in)
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

WHEEL = re.compile(r"^pet_kokkos-[^-]+\+(?P<variant>[a-z0-9]+)-.*\.whl$")


def page(title: str, links: list[str]) -> str:
    body = "\n".join(f"    {a}<br>" for a in links)
    return (
        "<!DOCTYPE html>\n<html>\n  <head><meta name=\"pypi:repository-version\" content=\"1.0\">"
        f"<title>{title}</title></head>\n  <body>\n{body}\n  </body>\n</html>\n"
    )


def main(repo: str, out: Path) -> None:
    lines = subprocess.check_output(["gh", "api", "--paginate", "--jq", ".[]", f"repos/{repo}/releases"], text=True)
    releases = [json.loads(line) for line in lines.splitlines() if line.strip()]
    wheels = defaultdict(dict)  # variant -> name -> link
    for rel in releases:
        if rel["draft"]:
            continue
        for a in rel["assets"]:
            m = WHEEL.match(a["name"])
            if not m:
                continue
            url = a["browser_download_url"]
            if a.get("digest", "").startswith("sha256:"):
                url += "#sha256=" + a["digest"].removeprefix("sha256:")
            wheels[m["variant"]][a["name"]] = (
                f'<a href="{html.escape(url)}" data-requires-python="&gt;=3.10">{html.escape(a["name"])}</a>'
            )
    for variant, files in sorted(wheels.items()):
        d = out / variant / "pet-kokkos"
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(page("Links for pet-kokkos", [files[n] for n in sorted(files)]))
        (out / variant / "index.html").write_text(page("Simple index", ['<a href="pet-kokkos/">pet-kokkos</a>']))
        print(f"{variant}: {len(files)} wheels")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(
        page("pet-kokkos GPU wheels", [f'<a href="{v}/">{v}</a>' for v in sorted(wheels)])
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], Path(sys.argv[2]))
