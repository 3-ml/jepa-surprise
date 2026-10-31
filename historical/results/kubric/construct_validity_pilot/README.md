# V-JEPA2-G Kubric construct-validity pilot

Eight accepted MOVi-A scenes share byte-identical 12-frame contexts. Four target-object hue doses are compared with four persistent keyed positive-z state displacements. Target hold and target shuffle are additional controls.

- `frozen_manifest.json` fixes scenes, objects, conditions, event frame, and rendered paths.
- `render_validation/` contains state, visibility, blend, context, and rendered-effect checks over 72 condition outputs.
- `model/` contains raw scores and the compact temporal-spatial error tensor.
- `analysis/` contains per-scene deltas, condition summaries, temporal summaries, and 10,000-draw scene-bootstrap comparisons.
- `review/` contains scene contact sheets and the visibility gate.
- `external_artifact_inventory.json` and `external_artifact_verification.json` describe the omitted large render tree and its independent checksum readback.

Target shuffle raises the clean score in all eight scenes. The matched physical-minus-appearance contrasts are 0.000259 [-0.000632, 0.001145] for 120-degree hue versus 0.25-diameter displacement and 0.000201 [-0.000791, 0.001035] for 180-degree hue versus 0.50-diameter displacement. The pilot does not establish equivalence or a population effect.
