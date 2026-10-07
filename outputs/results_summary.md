# Results summary: Plant Leaf Disease Detection (Group 38)

> **All results were obtained on the 150-images-per-class subset of PlantVillage (5,700 of 54,305 images,
> 38 classes), trained on CPU only.** They are not full-dataset results.


## Data
- Split (stratified, seed 42, saved in `outputs/splits/`): train 3,990 / val 855 / test 855 (70/15/15)
- Preprocessing for the main model: resize 260×260 → CLAHE on LAB L (clip 2.0, 8×8) → 3×3 median filter
- HSV disease segmentation (visualisation only): ROC AUC 0.799 for separating diseased from healthy leaves by estimated diseased area

## Model
- EfficientNetB2 (ImageNet, include_top=False) → GAP → Dense(128, ReLU) → Dropout(0.3) → Dense(38, softmax)
- Parameters: total 7,953,823; trainable in frozen phase 185,254 (head only)
- Training: Adam 1e-3 with frozen base (30 epochs), then top 30 layers unfrozen (BatchNorm frozen) with Adam 1e-5 (15 epochs); early stopping/checkpointing/LR schedule on validation loss
- Weights kept: phase2_finetune (best val loss 0.2546); training time 327.9 min on CPU

## Main model: held-out test set (single evaluation, 855 images)
| Metric | Value |
|---|---|
| Test accuracy | 93.22% (797/855) |
| Test loss | 0.2003 |
| Macro F1 | 0.9325 |

Lowest per-class F1: Tomato: Septoria leaf spot 0.694; Tomato: Late blight 0.718; Tomato: Target Spot 0.778; Tomato: Early blight 0.791; Tomato: Bacterial spot 0.800

Most frequent confusions: Corn (maize): Cercospora leaf spot Gray leaf spot → Corn (maize): Northern Leaf Blight (5); Tomato: Early blight → Tomato: Septoria leaf spot (3); Tomato: healthy → Tomato: Target Spot (3); Corn (maize): Common rust → Corn (maize): Northern Leaf Blight (2); Corn (maize): Northern Leaf Blight → Corn (maize): Cercospora leaf spot Gray leaf spot (2)

## Ablation: CLAHE + denoising (shorter schedule: ≤10 frozen + ≤5 fine-tune epochs)
Mean ± sample std over 3 seeds (42, 43, 44), same split for all runs.

| Arm | Runs (seeds) | Test accuracy | Test macro F1 | Test loss | Epochs (frozen+FT), mean | Train time / run (min) |
|---|---|---|---|---|---|---|
| with CLAHE+denoise | 3 (42, 43, 44) | 0.9154 ± 0.0086 | 0.9157 ± 0.0082 | 0.2542 ± 0.0163 | 15.0 | 31.0 |
| without (resize only) | 3 (42, 43, 44) | 0.9290 ± 0.0029 | 0.9287 ± 0.0031 | 0.2186 ± 0.0164 | 15.0 | 27.2 |
| Difference (with − without), paired by seed | 3 pairs | -0.0136 ± 0.0114 | -0.0130 ± 0.0112 | 0.0356 ± 0.0288 |  |  |

## Files
- Figures: `outputs/*.png`; per-epoch histories: `outputs/history/`; classification report: `outputs/classification_report.csv`
- Ablation: `outputs/ablation/ablation_runs.csv`, `outputs/ablation/ablation_summary.csv`
- Model: `models/efficientnetb2_plantvillage_subset150.keras` (not committed; `models/` is gitignored)

## Notes added after the run
- **Training time.** The machine went into system suspend twice during the run. Main-model compute was about 81 min. The 327.9 min above is wall-clock time and includes a suspended period of about 4.1 h during frozen epoch 22. The `with` arm's 31.0 min/run is inflated by a 13-min suspend in the seed-44 run; real compute is about 27 min/run for both arms.
- **Ablation interpretation.** CLAHE + median denoising lowered test accuracy on this subset for every seed (paired difference −1.36 ± 1.14 pp over 3 seeds). With 3 seeds and 855 test images this is evidence of no benefit, not proof of harm. The main model (enhanced inputs, longer schedule, 93.22%) is not directly comparable to either ablation arm.
