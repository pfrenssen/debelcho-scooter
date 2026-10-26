# SPDX-License-Identifier: GPL-3.0-or-later

"""Close-up inspection render: swaps in a camera that rides with the scooter.

Usage:
    blender -b build/cat_scooter.blend --python scripts/render_closeup.py \
        -o //preview/closeup_#### -F PNG -f 20 -- X Y Z TARGET_X TARGET_Z [nofur] [lit]

X Y Z is the camera position and (TARGET_X, 0, TARGET_Z) the point it looks at, both in the
scooter frame (+X forward, +Y left, +Z up, origin on the ground below the deck).
nofur hides the particle hair to show the body shape; lit adds a soft light at the camera.
"""

import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
x, y, z, tx, tz = map(float, args[:5])
scene = bpy.context.scene
rig = bpy.data.objects["ScooterRig"]

target = bpy.data.objects.new("CloseupTarget", None)
scene.collection.objects.link(target)
target.parent = rig
target.location = (tx, 0.0, tz)

data = bpy.data.cameras.new("Closeup")
data.lens = 50
cam = bpy.data.objects.new("Closeup", data)
scene.collection.objects.link(cam)
cam.parent = rig
cam.location = (x, y, z)
track = cam.constraints.new('TRACK_TO')
track.target = target
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'
scene.camera = cam
scene.render.use_motion_blur = False

if "nofur" in args:
    for obj in scene.objects:
        for mod in obj.modifiers:
            if mod.type == 'PARTICLE_SYSTEM':
                mod.show_render = False

if "lit" in args:
    light = bpy.data.lights.new("InspectLight", 'AREA')
    light.energy = 30.0
    light.size = 1.0
    lamp = bpy.data.objects.new("InspectLight", light)
    scene.collection.objects.link(lamp)
    lamp.parent = cam
    lamp.location = (0.3, 0.4, 0.0)
