# SPDX-License-Identifier: GPL-3.0-or-later

"""Switch the scene to Cycles for a final-quality render.

Usage:
    blender -b build/cat_scooter.blend --python scripts/render_cycles.py -o //render/cycles_#### -F PNG -f 240

Uses the GPU when Cycles finds one, otherwise the CPU. Adaptive sampling up to 128 samples
with OpenImageDenoise.
"""

import bpy

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
cycles = scene.cycles

prefs = bpy.context.preferences.addons['cycles'].preferences
for kind in ('OPTIX', 'CUDA', 'HIP', 'ONEAPI'):
    prefs.compute_device_type = kind
    prefs.refresh_devices()
    gpus = [d for d in prefs.devices if d.type != 'CPU']
    if gpus:
        for device in gpus:
            device.use = True
        cycles.device = 'GPU'
        break
else:
    prefs.compute_device_type = 'NONE'
    cycles.device = 'CPU'
print(f"Cycles device: {cycles.device}")

cycles.samples = 128
cycles.use_adaptive_sampling = True
cycles.adaptive_threshold = 0.03
cycles.use_denoising = True
cycles.denoiser = 'OPENIMAGEDENOISE'
cycles.transparent_max_bounces = 16  # stacked alpha-tested twig cards and grass
cycles.volume_step_rate = 4.0  # the haze is thin and smooth; coarse steps suffice
scene.render.use_persistent_data = True
