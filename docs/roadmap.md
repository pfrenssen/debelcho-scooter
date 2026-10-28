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
- **Chest fur through the kerchief.** Strands poke through the hanging triangle, most visibly
  in Cycles (the `chest` view). The fur is removed only where the skin is within about 5 mm of
  a cloth vertex (`FurDensity` in `build_cat`), but the chest fur is 19 mm long and the guard
  hairs 1.7 times that, so strands rooted round the cloth can still reach through it. Interpolated
  children can also land in the bare zone. Find which system pokes through first (render one
  particle system at a time); then shorten or comb away the strands near the cloth, or cut
  them where they meet it.
- **Notice the camera.** Around frame 260 the cat should briefly glance towards the camera on
  its left, as if it notices or acknowledges the camera's presence. This will break the fixed
  head position throughout the entire clip and give more life to the scene.
- **Small movements.** Apart from leaning forwards and backwards the cat is mostly motionless.
  It should move its left elbow at some point, for example after hitting a speed bump, and
  also make another small moment in another point, like hunching its shoulders a bit.

## Environment

- **Branch motion.** Grass clumps and firs lean as rigid wholes with the gusts. Bending
  trunks and fluttering branches need per-vertex deformation; doing that as shader displacement
  made the Radeon 680M time out and reset during animation renders.
- **Wheel interaction.** The scooter pushes the verge grass aside and raises a faint dust trail.
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
