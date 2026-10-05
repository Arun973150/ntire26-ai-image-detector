# Generalized AI-Image Detector (NTIRE 2026)

Answers one question: **is this image AI-generated or AI-edited?** The detector is built to generalize to generators it has never seen. It trains on the **NTIRE 2026 Robust AI-Generated Image Detection in the Wild** data (20 open-source generators) and is evaluated on held-out generators, including **Nano Banana**, Qwen-Image, HiDream, Grok Imagine, and SeeDream.

Nano Banana is never in the training set, so detecting it is an honest unseen-generator result.

---

## Approach

| Part | Choice |
|---|---|
| Backbone | DINOv3 ViT-L/16 (DINOv2-with-registers as an ungated fallback) |
| Head | CLS + 4 register tokens + mean-patch + attention pooling → MLP |
| Model ladder | frozen baseline → LoRA (r=16, all linear layers) → full fine-tune → 2-expert ensemble with flip TTA |
| Loss / optimizer | Focal loss (γ=2, α=0.5), AdamW with cosine warmup, bf16, gradient checkpointing at 512 px |
| Augmentation | Mirrors NTIRE's robust track: recompression, resize, blur, noise, and color jitter in severity buckets |

**Shortcut control.** Every image, real or fake, goes through the same decode → resize → crop → JPEG q90 funnel, so file format can't become the label. Turning `reencode_jpeg` off in a config retrains without it. A big gap between the two runs means the model was learning compression, not generator artifacts.

## Evaluation

- **Primary:** Clean and Robust ROC-AUC, using NTIRE's 36-transform robustness regime
- **Per held-out generator AUC:** the leave-generators-out scorecard
- **External Nano Banana test:** Pico-Banana-400K, kept strictly out of training

## Configs

| Config | What it trains |
|---|---|
| `phase1_dinov2.yaml` | Frozen DINOv2 baseline (no gated model needed) |
| `phase1_frozen.yaml` | Frozen DINOv3 baseline and normalization audit |
| `phase2_lora.yaml` | LoRA on DINOv3 at 512 px, the main workhorse |
| `phase2_lora_cls.yaml` | LoRA at 384 px with CLS + mean pooling, a decorrelated ensemble member |
| `phase2_lora_mixed.yaml` | LoRA on the mixed NTIRE + edit datasets, to cover AI edits as well as full generations |
| `phase_native.yaml` | Native-resolution 512 px crops, no re-encode, to keep high-frequency artifacts |
| `phase3_fullft.yaml` | Full fine-tune of the backbone |

## Quick start

```bash
pip install -r requirements.txt          # torch/torchvision come from your PyTorch + CUDA image
python -m scripts.check_backbone         # verify DINOv3 access and token layout
python -m src.download_ntire --shards 0 --out data/ntire
python -m src.prep_ntire --data-root data/ntire --out-dir data
python -m src.train --config configs/phase1_frozen.yaml
python -m src.eval --ckpt results/phase1_frozen/best.pt --manifest data/manifest_val.csv
```

Run the detector on your own images:

```bash
python -m src.predict --ckpt results/phase_native/best.pt --crops 8 photo1.jpg photo2.png
```

Smoke-test the whole loop on CPU without downloading anything:

```bash
python -m tests.make_synthetic
python -m src.prep_ntire --data-root data/ntire/shard_synthetic --out-dir data
python -m src.train --config configs/phase1_frozen.yaml
```

Training on RunPod (A100): see [SETUP.md](SETUP.md), or run `bash runpod_quickstart.sh`. The full research plan, with phases, gates, and risks, is in [PLAN.md](PLAN.md).

## Layout

| Path | Purpose |
|---|---|
| `src/download_ntire.py`, `src/prep_*.py` | Download NTIRE shards; build manifests for NTIRE, Pico-Banana, and Hugging Face edit datasets |
| `src/build_mixed.py` | Merge source manifests into one balanced train/val set |
| `src/models.py`, `src/losses.py` | DINOv3 detector with pooling heads; focal loss |
| `src/train.py`, `src/eval.py`, `src/ensemble_eval.py` | Training, per-generator evaluation, ensemble + TTA evaluation |
| `src/predict.py` | Inference on image files |
| `src/compact_shard.py` | Shrink shards to a 512 px cache so all six fit in about 24 GB |
| `probe/derive_masks.py` | Tests whether edit masks can be derived from Pico-Banana pairs |

## Access and licenses

DINOv3 and the NTIRE training set are gated on Hugging Face, so accept their terms first. Pico-Banana-400K is CC BY-NC-ND. Use this project for research only, and don't redistribute the data.
