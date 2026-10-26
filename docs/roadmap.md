# Roadmap

Future improvements to the scene, roughly in order of how much each one adds to the realism of
the shot.

## Cat

- **Attached ears.** The ears look detached from the head. They are separate objects that
  rotate for the twitches and the wind flutter; blend them into the head so no gap or seam shows
  at the base in any pose.
- **Fur in Cycles.** In Cycles renders the cat is a flat black silhouette: the strands use a
  Principled BSDF, which shows neither the fur texture nor light from behind. Switch the strand
  material to the Principled Hair BSDF (melanin parametrisation, near-black) for Cycles, so the
  fur shows sheen and a rim against dark backgrounds. EEVEE 5.2 renders that BSDF black, so the
  render scripts need to select the strand shader per engine.
- **Fur grooming.** Vary the length by region (shorter on the face and paws, longer on the chest
  and tail), add clumping and some flyaway strands.
- **Toes on the grips.** The front toes merge into one shape on the handlebar grip; separate
  them so each toe wraps the grip.

## Environment

- **Branch motion.** Grass clumps and firs lean as rigid wholes with the gusts. Bending
  trunks and fluttering branches need per-vertex deformation; doing that as shader displacement
  made the Radeon 680M time out and reset during animation renders.
- **Wheel interaction.** The scooter pushes the verge grass aside and raises a faint dust trail.
- **Worn road markings.** The centre line dashes are a clean, uniform `RoadPaint` material and
  look freshly painted. Mountain roads are rarely repainted: make the markings heavily worn and
  faded, with the asphalt showing through, chipped edges, and more wear where the tyres run.
- **Road detail.** Painted edge lines, a gravel shoulder, worn tyre tracks, repair patches, a
  slight crown, and grit along the edges where the road meets the grass.
- **Terrain erosion.** Run a hydraulic erosion pass on the near height grid for gullies, ridges
  and smoother valley floors.
- **Vegetation variety.** Bushes, wildflowers and fallen needles on the verges; a second tree
  species for variety next to the firs.

## Rendering

- **Cycles on the GPU.** With `hip-runtime-amd` installed, Cycles lists the Radeon 680M as a HIP
  device, but a frame renders no faster than on the 16-thread CPU (about 2 min 15 s) and still
  uses about 34 min of CPU time. Find out whether the render actually runs on the GPU.
- **Final render in Cycles.** At about 2 min 15 s per frame, the 288-frame animation takes
  about 10.5 hours on the CPU.
