"""Quick Workbench wireframe render of the animation, to check motion.

Usage:
    blender -b build/cat_scooter.blend --python scripts/render_wireframe.py -a

Writes build/render/wireframe_0001-NNNN.mkv, which plays while it is still being written. Fur is hidden. Workbench does not render the
WIREFRAME shading type, so the scooter and the cat get a Wireframe modifier and the
environment is shaded solid.
"""

import bpy

scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
shading = scene.display.shading
shading.type = 'SOLID'
shading.light = 'FLAT'
shading.color_type = 'OBJECT'
scene.render.use_motion_blur = False
scene.render.filepath = "//render/wireframe_"
scene.render.ffmpeg.format = 'MKV'  # unlike MP4, it has no index written only at the end
scene.world.color = (0.02, 0.02, 0.025)

for obj in scene.objects:
    for mod in obj.modifiers:
        if mod.type == 'PARTICLE_SYSTEM':
            mod.show_render = False

for obj in bpy.data.collections["Actors"].objects:
    if obj.type == 'MESH':
        wire = obj.modifiers.new("Wire", 'WIREFRAME')
        wire.thickness = 0.0012
        wire.use_even_offset = False

# Colour the moving parts so they stand out from the terrain.
colours = {
    "Spring": (1.0, 0.8, 0.0, 1.0),
    "Shock": (1.0, 0.8, 0.0, 1.0),
    "Wheel": (0.2, 0.8, 1.0, 1.0),
    "Fork": (0.2, 0.8, 1.0, 1.0),
    "RearArm": (0.2, 0.8, 1.0, 1.0),
    "Scooter": (1.0, 0.4, 0.1, 1.0),
    "Cat": (1.0, 1.0, 1.0, 1.0),
    "Ear": (1.0, 1.0, 1.0, 1.0),
    "Eye": (1.0, 1.0, 1.0, 1.0),
}
for obj in scene.objects:
    obj.color = next((c for key, c in colours.items() if key in obj.name), (0.25, 0.3, 0.25, 1.0))
