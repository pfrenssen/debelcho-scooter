"""Download the CC0 Poly Haven textures the scene uses into assets/textures/.

Usage:
    python3 scripts/fetch_textures.py

Each texture is saved as assets/textures/<asset>/{diff,nor_gl,rough}.jpg at 2k.
Existing files are kept.
"""

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXTURES = ROOT / "assets" / "textures"
ASSETS = ["grass_ground", "dry_ground_01", "aerial_rocks_02", "asphalt_02"]
MAPS = {"Diffuse": "diff", "nor_gl": "nor_gl", "Rough": "rough"}
RESOLUTION = "2k"
# The Poly Haven API rejects the default urllib User-Agent and asks for an identifying one.
HEADERS = {"User-Agent": "debelcho-scooter-texture-fetch/1.0"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS)) as response:
        return response.read()


def main():
    for asset in ASSETS:
        target = TEXTURES / asset
        target.mkdir(parents=True, exist_ok=True)
        files = json.loads(get(f"https://api.polyhaven.com/files/{asset}"))
        for key, name in MAPS.items():
            path = target / f"{name}.jpg"
            if path.exists():
                continue
            url = files[key][RESOLUTION]["jpg"]["url"]
            print(f"{asset}/{name}.jpg <- {url}")
            path.write_bytes(get(url))


if __name__ == "__main__":
    main()
