# SPDX-License-Identifier: GPL-3.0-or-later

"""Report the smallest gap between the cat's head and the handlebar over the animation.

Usage:
    blender -b build/cat_scooter.blend --python scripts/check_clearance.py

The handlebar is the bar tube with its grips plus the dashboard on top. The gap is measured from
the head surface (without fur) and is negative where the head passes through.
"""

import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_scene as bs  # noqa: E402  (constants only; main() runs only as __main__)

scene = bpy.context.scene
head = bpy.data.objects["CatHead"]
frame_obj = bpy.data.objects["ScooterFrame"]  # its local frame is the scooter frame
depsgraph = bpy.context.evaluated_depsgraph_get()
bar = np.array(bs.BAR)
dash_centre, dash_half = bar + np.array((0.006, 0.0, 0.006)), np.array((0.018, 0.0275, 0.01))
grip_radius = 0.0088

gaps = []
for frame in range(1, scene.frame_end + 1, 2):
    scene.frame_set(frame)
    evaluated = head.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    evaluated.to_mesh_clear()
    to_scooter = np.array(frame_obj.matrix_world.inverted() @ head.matrix_world)
    co = co.reshape(-1, 3) @ to_scooter[:3, :3].T + to_scooter[:3, 3]

    along = np.abs(co[:, 1]) <= bs.BAR_HALF_WIDTH + 0.008
    tube = np.where(along, np.linalg.norm(co[:, [0, 2]] - bar[[0, 2]], axis=1) - grip_radius, np.inf)
    q = np.abs(co - dash_centre) - dash_half
    dash = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
    gaps.append((min(tube.min(), dash.min()), frame))

for gap, frame in sorted(gaps)[:5]:
    print(f"CLEARANCE frame {frame}: {gap * 1000:.1f} mm")
