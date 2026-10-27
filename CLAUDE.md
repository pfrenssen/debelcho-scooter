# Working rules

- Commit after every completed and verified step.
- Never start a full-quality render of the whole clip without asking first, even after an
  earlier one was approved. To show a change, render stills or a quick preview (50 %
  resolution, 16 samples, no motion blur) into `build/preview/`, and give their paths in the
  report. `scripts/preview_views.sh` renders named close-up views at that quality; add a view
  to `VIEWS` in `render_closeup.py` when a new one is needed.
- Run every Blender command under `timeout`, sized to the expected duration with some margin.
  In scripts that loop over frames, compute per-frame data once per frame: `scene.frame_set`
  re-evaluates the geometry nodes of the grass and the trees, so calling it in an inner loop
  runs for many minutes.
