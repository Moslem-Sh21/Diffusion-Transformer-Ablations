# Diffusion-Transformer-Ablations


**Diffusion-Transformer-Ablations** is a research toolkit for systematic ablation and interpretability analysis in diffusion models with DiT (Diffusion Transformer) backbones, such as PixArt-α.  
It enables researchers to selectively disable (ablate) self-attention, cross-attention, and MLP submodules in any DiT block during inference, and to quantify their importance via FID and CLIP metrics.

---

## Features

- **Ablate any block or submodule** (self-attention, cross-attention, MLP) in a DiT-based transformer.
- **Supports PixArt-α (Diffusers) out of the box.**
- **Batch generation** of images from COCO prompts.
- **Automated evaluation**: compute FID (with torch-fidelity) and CLIP similarity scores for generated images.

---

## Usage

1. **Prepare**:  
   - Download COCO val2017 images and captions.
   - Install dependencies: `diffusers`, `torch-fidelity`, `openai-clip`, `torch`, `PIL`, `tqdm`.

2. **Edit ablation settings** in the main script:  
   Choose which layers and which submodules (`attn1`, `attn2`, `ff`) to ablate, and set ablation mode (`zero`, `input`, or `mean`).

3. **Run** the script to:
   - Generate N images with and without ablation.
   - Compute FID and CLIP metrics for comparison.

---

## Example

```python
# Ablate self-attention and MLP in the last block
ablate_dit_parts(
    pipe.transformer, blocks_to_patch=[11],
    parts=('attn1', 'ff'), mode='input'
)



