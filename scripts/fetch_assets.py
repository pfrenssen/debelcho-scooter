# SPDX-License-Identifier: GPL-3.0-or-later

"""Download the CC0 Poly Haven textures and models the scene uses into assets/.

Usage:
    python3 scripts/fetch_assets.py

Textures are saved as assets/textures/<asset>/{diff,nor_gl,rough}.jpg at 2k. Models are saved
as assets/models/<asset>/<asset>.blend with the textures it references next to it.
Existing files are kept.
"""

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXTURES = ROOT / "assets" / "textures"
MODELS = ROOT / "assets" / "models"
TEXTURE_ASSETS = ["grass_ground", "dry_ground_01", "aerial_rocks_02", "asphalt_02"]
MODEL_ASSETS = ["fir_tree_01"]
MAPS = {"Diffuse": "diff", "nor_gl": "nor_gl", "Rough": "rough"}
TEXTURE_RESOLUTION = "2k"
MODEL_RESOLUTION = "1k"
# The Poly Haven API rejects the default urllib User-Agent and asks for an identifying one.
HEADERS = {"User-Agent": "debelcho-scooter-texture-fetch/1.0"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS)) as response:
        return response.read()


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{path.relative_to(ROOT)} <- {url}")
    path.write_bytes(get(url))


def files(asset):
    return json.loads(get(f"https://api.polyhaven.com/files/{asset}"))


def main():
    for asset in TEXTURE_ASSETS:
        listing = files(asset)
        for key, name in MAPS.items():
            download(listing[key][TEXTURE_RESOLUTION]["jpg"]["url"], TEXTURES / asset / f"{name}.jpg")
    for asset in MODEL_ASSETS:
        blend = files(asset)["blend"][MODEL_RESOLUTION]["blend"]
        target = MODELS / asset
        download(blend["url"], target / f"{asset}.blend")
        for relative, include in blend.get("include", {}).items():
            download(include["url"], target / relative)


if __name__ == "__main__":
    main()
