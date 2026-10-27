# Roadmap

Future improvements to the scene, roughly in order of how much each one adds to the realism of
the shot.

## Cat

- **Sun on the climb.** From frame 85 to 165 the cat climbs a stretch steeper than the 4° sun,
  so the road ahead shades it. Easing that grade, or a light linked to the cat for those frames,
  would keep it in the sun.
- **Fur grooming.** Vary the length by region (shorter on the face and paws, longer on the chest
  and tail), add clumping and some flyaway strands.
- **Toes on the grips.** The front toes merge into one shape on the handlebar grip; separate
  them so each toe wraps the grip.

## Environment

The forest and fog items below are steps towards the look of
`~/Downloads/Austria-kleinwalsertal-hiking3.jpg` (not in the repository): hills under a dense fir
forest, paler and greyer with distance, with low cloud caught on the slopes.

- **Distance fog.** A shared node group that fades every material (terrain, trees, barn, road)
  towards a fog colour with camera distance, more at low height, as `alpine_material` already
  does for the far terrain. It gives the layered silhouettes that grow paler with distance.
- **Fog banks.** Low cloud drifting over the slopes: stretched volume ellipsoids within the
  300 m EEVEE volume range, their density from 4D noise × a height falloff × a soft edge, with
  the noise W animated for the drift.
- **Branch motion.** Grass clumps and firs lean as rigid wholes with the gusts. Bending
  trunks and fluttering branches need per-vertex deformation; doing that as shader displacement
  made the Radeon 680M time out and reset during animation renders.
- **Wheel interaction.** The scooter pushes the verge grass aside and raises a faint dust trail.
- **Road detail.** A gravel shoulder, worn tyre tracks, repair patches and a slight crown.
- **Terrain erosion.** Run a hydraulic erosion pass on the near height grid for gullies, ridges
  and smoother valley floors.
- **Vegetation variety.** Bushes, wildflowers and fallen needles on the verges; a second tree
  species for variety next to the firs.

## Rendering

- **Cycles on the GPU.** `render_cycles.py` enables the GPUs but leaves the CPU device on, so
  every render so far ran on CPU and GPU together. On the laptop (Ryzen 7 PRO 6850U, Radeon
  680M over HIP) a 1280×720 frame takes about 2 min 15 s and about 34 min of CPU time, no faster
  than the CPU alone; a 1920×1200 frame takes 5 to 7 min. On the workstation (Ryzen 9 3950X,
  GTX 1060 6GB) a 1280×720 frame takes 65 s with 27 min of CPU time. Time a GPU-only and a
  CPU-only frame on each machine to find the fastest device set.
- **Final render in Cycles.** The 480-frame clip takes about 8.7 hours on the workstation and
  about 18 hours on the laptop, measured before the forest and the grass culling.
