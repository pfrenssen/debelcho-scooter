# Debelcho on a scooter

An experiment in vibe coding a Blender scene: a black cat riding an electric scooter along an
alpine road at sunset. The whole scene (terrain, road, barn, cat, scooter, animation, lighting
and grading) is built procedurally by Python scripts written by an AI coding agent from
conversational prompts. No modelling was done by hand.

## Requirements

Blender 5.2, Python 3 and ffmpeg.

## Build and render

```bash
python3 scripts/fetch_assets.py                                 # Poly Haven textures and firs into assets/
blender -b --factory-startup --python scripts/build_scene.py   # writes build/cat_scooter.blend
blender -b build/cat_scooter.blend -a                           # EEVEE, MP4 in build/render/
```

For Cycles, render a PNG sequence and encode it:

```bash
blender -b build/cat_scooter.blend --python scripts/render_cycles.py -o //render/cycles_#### -F PNG -a
ffmpeg -framerate 24 -i build/render/cycles_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 cycles.mp4
```

Planned improvements are in [docs/roadmap.md](docs/roadmap.md).

## Licence

GPL-3.0-or-later, see [LICENSE](LICENSE).
