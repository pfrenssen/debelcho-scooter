# Roadmap

Future improvements to the scene, roughly in order of how much each one adds to the realism of
the shot.

## Cat

- **Secondary motion.** The rig drives the pelvis and spine only. Add ear and whisker flutter
  in the wind, a breathing cycle, and a short head bob that lags the bumps.
- **Fur grooming.** Vary the length by region (shorter on the face and paws, longer on the chest
  and tail), add clumping and some flyaway strands, and use the Principled Hair BSDF for Cycles
  renders.
- **Face detail.** Eyes with a refractive cornea over the iris, tufts inside the ears, a
  mouth line and whisker pads.
- **Toes on the grips.** The front toes merge into one shape on the handlebar grip; separate
  them so each toe wraps the grip.

## Environment

- **Wind.** Grass and trees sway with slow gusts travelling across the hills, instead of
  standing frozen.
- **Wheel interaction.** The scooter pushes the verge grass aside and raises a faint dust trail.
- **Road detail.** Painted edge lines, a gravel shoulder, worn tyre tracks, repair patches, a
  slight crown, and grit along the edges where the road meets the grass.
- **Terrain erosion.** Run a hydraulic erosion pass on the near height grid for gullies, ridges
  and smoother valley floors.
- **Vegetation variety.** Bushes, wildflowers and fallen needles on the verges; a second tree
  species for variety next to the firs.

## Camera

- Light handheld shake, a slightly longer lens for the chase shots, focus that follows the cat,
  and subtle vignetting and film grain.

## Rendering

- **Cycles on the GPU.** With `hip-runtime-amd` installed, Cycles lists the Radeon 680M as a HIP
  device, but a frame renders no faster than on the 16-thread CPU (about 2 min 15 s) and still
  uses about 34 min of CPU time. Find out whether the render actually runs on the GPU.
- **Final render in Cycles.** At about 2 min 15 s per frame, the 288-frame animation takes
  about 10.5 hours on the CPU.
