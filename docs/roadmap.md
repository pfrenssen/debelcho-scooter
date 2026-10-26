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
- **Road edges.** The asphalt ends in a clean straight cut against the grass. Model the edges
  on an Austrian alpine pass road (reference photo: `~/Downloads/gettyimages-1195083867-612x612.jpg`,
  not in the repository):
  - on the valley side, a row of rough-hewn granite guard stones, about 0.6 m tall and 0.3 m
    square, every 2 to 3 m, each set slightly askew, replacing the wire fence;
  - on the uphill side, a low kerb or shallow concrete drainage gutter, with an occasional stone
    drain inlet;
  - a solid white edge line a few centimetres in from each edge, worn like the centre line;
  - a ragged asphalt edge with a dark rim of dirt where the grass grows over it.
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
  about 18 hours on the laptop. It needs **Fur in Cycles** first: with the current strand
  shader the cat renders as a flat black silhouette.
