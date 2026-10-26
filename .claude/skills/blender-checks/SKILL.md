---
name: blender-checks
description: Blender 5.2 gotchas for this scene. Use before running a Blender script or render from the command line, building shader or compositor nodes, adding lights or changing the fur, or judging a change from rendered stills.
---

# Blender checks

Gotchas found the hard way in this project, on Blender 5.2.3. Each one cost a wasted round.

## Running Blender

- Pass `--python-exit-code 1` with every `--python`. Without it, Blender carries on rendering
  after the script raises, and the output looks like a valid result of a script that never ran.
- Check a scratch script's own output line (e.g. a `print` of the values it set) in the log
  before judging the images.

## Nodes

- Compositor Color Balance: the type is `'Offset/Power/Slope (ASC-CDL)'`. Its sockets come in
  pairs with the same name; address them by identifier: `Base Offset` (scalar, takes negative
  values) and `Color Offset` (RGBA, clamps below 0), likewise `Color Slope`, `Color Power`.
- A Noise Texture `Fac` spreads only about ±0.06 round 0.5. Before thresholding it, stretch it
  (Math `MULTIPLY_ADD`, e.g. ×4 −1.5), so the threshold gives partial coverage instead of
  all-or-nothing.

## Lights and fur

- Light linking (`light_linking.receiver_collection`) limits surfaces only: the light still
  scatters in the haze volume. Set `light.volume_factor = 0` on a linked light.
- The fur is STRIP hair with a Principled BSDF: strip normals face the camera, so back light
  shows as random glints, not a rim. The Principled Hair BSDF renders black in EEVEE; it works
  in Cycles only.

## Judging stills

- Motion blur and depth of field hide surface detail in shot frames. Judge materials and
  geometry on a static close camera without DOF or motion blur, then confirm in a shot frame.
- Measure white balance on a neutral surface (an asphalt patch), not on whole-frame averages:
  grass and terrain albedo skew those.
