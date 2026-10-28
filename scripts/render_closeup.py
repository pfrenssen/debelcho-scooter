"""Close-up inspection render: swaps in a camera that rides with the scooter.

Usage:
    blender -b build/cat_scooter.blend --python scripts/render_closeup.py \
        -o //preview/closeup_#### -F PNG -f 20 -- VIEW [nofur] [lit] [preview]
    blender -b build/cat_scooter.blend --python scripts/render_closeup.py \
        -o //preview/closeup_#### -F PNG -f 20 -- X Y Z TARGET_X TARGET_Z [nofur] [lit] [preview]

VIEW is one of the named views in VIEWS. X Y Z is the camera position and (TARGET_X, 0,
TARGET_Z) the point it looks at, both in the scooter frame (+X forward, +Y left, +Z up, origin
on the ground below the deck). nofur hides the particle hair to show the body shape; lit adds a
soft light at the camera; preview sets the preview quality of CLAUDE.md (50 % resolution,
16 samples). For Cycles, pass scripts/render_cycles.py before this script, so preview sets
the samples after it.
"""

import sys

import bpy

# Camera position and target (x, z) in the scooter frame.
VIEWS = {
    "head-q": (0.55, 0.2, 0.66, 0.13, 0.6),  # whole head, three-quarter from the left
    "head-side": (0.14, 0.42, 0.64, 0.13, 0.6),
    "head-front": (0.5, 0.0, 0.64, 0.13, 0.6),
    "head-near": (0.4, 0.13, 0.64, 0.14, 0.6),  # face fills the frame
    "eyes": (0.5, 0.18, 0.63, 0.15, 0.605),  # both eyes, from the side the cat looks to at frame 300
    "glance": (0.325, 0.567, 0.6, 0.13, 0.6),  # from where the cat looks at frame 270, in its glance
    "ears-q": (0.4, 0.12, 0.72, 0.11, 0.69),  # into the cups
    "ears-back": (-0.05, 0.3, 0.74, 0.11, 0.69),
    "body-side": (0.0, 1.6, 0.42, 0.0, 0.4),  # whole cat from the left
    "flank": (0.0, 0.45, 0.42, -0.02, 0.38),  # fur texture on the left side
    "chest":(0.55, 0.2, 0.62, 0.1, 0.5),
    "paws": (0.42, 0.42, 0.62, 0.22, 0.53),  # left front paw on the grip
    "grip": (0.336, 0.264, 0.495, 0.2, 0.4),  # toes round the left grip, from the front and above
    "elbow": (0.85, 0.3, 0.32, 0.1, 0.38),  # both arms from the front and below, for the elbow flare at frame 355
    "shoulders": (-0.9, 0.5, 0.75, 0.05, 0.5),  # from behind, for the shoulder hunch at frame 125
    "tail": (-0.6, 0.4, 0.5, -0.2, 0.4),
}

args = sys.argv[sys.argv.index("--") + 1:]
if args[0] in VIEWS:
    x, y, z, tx, tz = VIEWS[args[0]]
else:
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

if "preview" in args:
    scene.render.resolution_percentage = 50
    scene.eevee.taa_render_samples = 16
    scene.cycles.samples = 16

print(f"CLOSEUP camera=({x}, {y}, {z}) target=({tx}, 0, {tz}) engine={scene.render.engine} "
      f"{scene.render.resolution_percentage}% options={[a for a in args if a in ('nofur', 'lit', 'preview')]}")
