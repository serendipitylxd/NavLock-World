# Model Notes

This repository does not include model weights.

## VLM Semantic Branch

The main semantic branch in the paper uses Qwen3-VL with LoRA adaptation. Put
local model weights under `models/` and update the corresponding YAML config:

```text
models/Qwen3-VL-4B-Instruct/
models/Qwen3-VL-8B-Instruct/
```

Example configs:

```text
configs/qwen3vl_4b_lora_unified.yaml
configs/qwen3vl_8b_lora_unified.yaml
configs/internvl3_5_4b_lora_unified.yaml
```

## Perception Backends

The paper uses:

- RTMDet for 2D ship detection;
- Hydro3DNet-style 3D detection for vessel geometry.

External OpenMMLab and detector dependencies should be installed separately.
