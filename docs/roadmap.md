# Roadmap

Future improvements to the scene, roughly in order of how much each one adds to the realism of
the shot.

## Cat

The face reads as a cartoon. The items below take it apart one feature at a time; judge each
on a close-up still in Cycles as well as EEVEE, since the fur strands shade differently in the two.

- **Sun on the climb** (low priority: the cat is only briefly in shade, right after the sun
  sets behind the hill). From frame 85 to 165 the cat climbs a stretch steeper than the 4° sun,
  so the road ahead shades it. Easing that grade, or a light linked to the cat for those frames,
  would keep it in the sun.

## Environment

- **Branch motion.** Grass clumps and firs lean as rigid wholes with the gusts. Bending
  trunks and fluttering branches need per-vertex deformation; doing that as shader displacement
  made the Radeon 680M time out and reset during animation renders.
- **Wheel interaction** (only if the line is moved onto the verge). The scooter rides on
  asphalt about 0.9 m from the nearest grass (`LANE_OFFSET`), so nothing in the shot causes
  pushed grass or dust. If a stretch of the line runs onto the verge, the clumps near the wheels
  bend away and the dirt raises a faint dust trail.
- **Road detail.** A gravel shoulder, a slight crown, and cracks in the geometry (the cracks
  now come only from the asphalt texture).
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
  about 18 hours on the laptop, measured before the forest and the grass culling. The fog
  banks nearly doubled a 50 %, 16-sample preview frame on the laptop GPU (10 s to 18 s).
