# Kubric result map

- `rendering_gate/` compares Cycles rendering settings on an A40 and records the selected 64-sample OptiX configuration and repeatability bound.
- `construct_validity_pilot/` contains the eight-scene V-JEPA2-G appearance-versus-state-displacement study, its validation records, compact score tensors, and visual review sheets.

The raw 1.188 GB render and simulator-state tree is not included. Its file count, byte count, checksum-manifest hash, and independent readback result are retained in the pilot directory.
