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

The scene renders at 1280×720. For 1080p or 2160p, set the resolution with
`scripts/render_resolution.py`:

```bash
blender -b build/cat_scooter.blend --python scripts/render_resolution.py -a -- 1920 1080
blender -b build/cat_scooter.blend --python scripts/render_resolution.py -a -- 3840 2160
```

For Cycles, render a PNG sequence and encode it (add `--python scripts/render_resolution.py`
and `-- 1920 1080` at the end for 1080p):

```bash
blender -b build/cat_scooter.blend --python scripts/render_cycles.py -o //render/cycles_#### -F PNG -a
ffmpeg -framerate 24 -i build/render/cycles_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 cycles.mp4
```

### Resuming an interrupted render

An interrupted MP4 render cannot be resumed, and the partial file does not play. For a long
render, write a PNG sequence with overwriting turned off. Running the same command again skips
every frame that already exists ("Skipping existing frame" in the log) and renders the rest:

```bash
blender -b build/cat_scooter.blend \
    --python-expr "import bpy; bpy.context.scene.render.use_overwrite = False" \
    -o //render/frames_#### -F PNG -a
```

The same `--python-expr` works with the Cycles and resolution scripts. Put it before `-o`,
because Blender processes its arguments in order. If Blender was killed while it was writing a
file, the newest PNG can be truncated, so delete it before you resume:

```bash
rm "$(ls -t build/render/frames_*.png | head -1)"
```

When every frame is done, encode the sequence with the `ffmpeg` command above.

Planned improvements are in [docs/roadmap.md](docs/roadmap.md).

### Setting the number of samples

For a quick test render, reduce the resolution and the number of samples.

```bash
blender -b build/cat_scooter.blend --python scripts/render_cycles.py -o //render/cycles_#### -F PNG --python scripts/render_resolution.py --python-expr "import bpy; bpy.context.scene.render.use_overwrite = False; bpy.data.scenes['Scene'].cycles.samples = 8" -a -- 1280 720
```

## Licence

GPL-3.0-or-later, see [LICENSE](LICENSE).
