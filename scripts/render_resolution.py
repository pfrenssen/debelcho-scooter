"""Set the output resolution for a render, e.g. 1920×1080 or 3840×2160.

Usage:
    blender -b build/cat_scooter.blend --python scripts/render_resolution.py -a -- 1920 1080

Combines with scripts/render_cycles.py for a Cycles render at that resolution. The scene
itself renders at 1280×720.
"""

import sys

import bpy

width, height = map(int, sys.argv[sys.argv.index("--") + 1:][:2])
render = bpy.context.scene.render
render.resolution_x, render.resolution_y = width, height
render.resolution_percentage = 100
print(f"Resolution: {width}×{height}")
