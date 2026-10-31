# Extraction Provenance

- Exact checkpoint SHA-256: `67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e`.
- V-JEPA2 source revision: `204698b45b3712590f06245fbfba32d3be539812`.
- Strategy: exact sequential CPU/GPU block offload, BF16 autocast, FP32 weights, no quantization.
- New valid jobs: 912; reused corrected samples: 608.
- Frames/resolution: frozen 64 / 256x256; five pooled paper layers only.
- Repeat checks: `[{"job_id": "vjepa2-g__e9e46d920dde8614ef2a9ab7631f980d4b6c7969673517b136f69e921ac071a1__selective_recolor_045", "max_abs_diff": 0.0}, {"job_id": "vjepa2-g__e9e46d920dde8614ef2a9ab7631f980d4b6c7969673517b136f69e921ac071a1__selective_recolor_090", "max_abs_diff": 0.0}, {"job_id": "vjepa2-g__ed246c85930f35ccf07f06a4248d583eaa8da966d0fcc0ae551e3c0f6f12b880__hue_015", "max_abs_diff": 0.0}]`.
- Packed feature SHA-256: `cc5c9a5c454b37a676816d6338019190724c7eeecf63965407a717582d3f83a4`.
