# Representation Validation

The first newly executed clean representation was compared against the corrected development V-JEPA2-G cache before full extraction was authorized.

| Layer | Cosine(new, cached) | Max abs diff | Mean abs diff |
|---|---:|---:|---:|
| block_00 | 1.00000012 | 0.008705 | 0.000310 |
| block_10 | 1.00000000 | 0.127930 | 0.000379 |
| block_20 | 0.99999988 | 0.233398 | 0.000415 |
| block_30 | 0.99999988 | 0.094849 | 0.000257 |
| final | 1.00000000 | 0.003788 | 0.000140 |

All layers pass the predeclared cosine gate of 0.999. The checkpoint, frame indices, transform implementation, normalization, layer semantics, and pooling match the corrected cache provenance. Repeat-determinism checks are recorded by the full extractor and will be summarized after completion.
