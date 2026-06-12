# wearable-stress-calibration

**Calibration, not architecture, limits cross-subject wearable stress detection: a multi-dataset, decision-focused evaluation with label-efficient recalibration.**

This repository contains the code, the leave-one-subject-out (LOSO) prediction files, and the figure-generation scripts that reproduce every figure and table in the paper:

> Ahmet Akkaya (2026). Manuscript under review.

The central finding is that on real-world wearable data a model can reach acceptable discrimination yet remain badly miscalibrated under subject shift, so that it offers **no clinical net benefit until it is calibrated to the individual**. A low-burden, few-shot (10–40 labels) per-subject recalibration recovers calibration and expands the range of decision thresholds over which the model is useful. Deeper or self-supervised models do **not** beat a calibrated feature + gradient-boosting baseline.

---

## Repository structure

```
wearable-stress-calibration/
├── config.py            # paths, hyperparameters, global seed (1337)
├── data_wesad.py        # WESAD loading + 60 s windowing + feature extraction
├── exstress.py          # Exercise-Stress (Hongn 2025) loading + features
├── baselines.py         # feature + gradient-boosting baseline
├── models.py            # compact 1D-CNN (scratch + self-supervised variants)
├── nurse_deep.py        # deep / SSL pathway for the Nurse dataset
├── calibration.py       # ECE, Brier, calibration slope/intercept, recalibration
├── decision_curve.py    # decision-curve analysis (net benefit)
├── stats_tests.py       # DeLong, McNemar, Spiegelhalter z-test
├── run_experiment.py    # orchestrates WESAD / Exercise feature-GB LOSO runs
├── run_nurse.py         # orchestrates the Nurse experiment (feature + deep)
├── compute_stats.py     # reads csv/ and prints all paper numbers and tables
├── make_figures.py      # reads csv/ and regenerates all figures
├── csv/                 # saved LOSO predictions and intermediate outputs
└── figures/             # generated publication figures
```

---

## Requirements

Python ≥ 3.9 with:

```
numpy
pandas
scipy
scikit-learn
torch
matplotlib
```

```bash
pip install numpy pandas scipy scikit-learn torch matplotlib
```

All experiments were run on Kaggle (single T4 GPU) with a fixed global seed of **1337**.

---

## Datasets

The raw datasets are **not redistributed here**. Download them from their original sources and set the paths in `config.py`.

| Dataset | Subjects used | Setting | Source |
|---|---|---|---|
| WESAD | 15 | Lab (TSST) | Schmidt et al. 2018 — UCI: https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection |
| Nurse Stress | 15 | Real-world hospital, Empatica E4 | Hosseini et al. 2022 — Kaggle: https://www.kaggle.com/datasets/priyankraval/nurse-stress-prediction-wearable-sensors |
| Exercise-Stress | 18 (v1 cohort) | Induced stress + exercise, Empatica E4 | Hongn et al. 2025 — PhysioNet: https://physionet.org/content/wearable-device-dataset/1.0.1/ (DOI 10.13026/he0v-tf17) |

All signals are processed as 60 s non-overlapping windows with robust (median/IQR) per-subject normalization and strict LOSO validation.

---

## How to reproduce

The `csv/` folder ships with the saved LOSO predictions, so you can reproduce **every number and figure without re-running the heavy experiments or downloading the raw data**.

**1. Reproduce all tables and statistics**

```bash
python compute_stats.py
```

Prints per-subject and pooled AUROC with confidence intervals, calibration metrics (ECE, Brier, slope, intercept), the few-shot calibration curve, the deep-vs-baseline comparison with DeLong / McNemar tests, and the decision-curve summary.

**2. Regenerate all figures**

```bash
python make_figures.py
```

Writes the figures into `figures/`.

**3. (Optional) Re-run the experiments from raw data**

After downloading the datasets and setting paths in `config.py`:

```bash
python run_experiment.py   # WESAD and Exercise-Stress feature-GB LOSO
python run_nurse.py        # Nurse feature + deep / SSL pathways
```

This regenerates the prediction files in `csv/`.

---

## What is in `csv/`

| File | Contents |
|---|---|
| `feat_loso_predictions.csv` | WESAD feature-GB LOSO predictions |
| `nurse_deep_pred.csv` | Nurse predictions: GB, self-supervised, from-scratch |
| `exstress_loso_pred.csv` | Exercise-Stress feature-GB LOSO predictions |
| `fewshot_calib_curve.csv` | Nurse few-shot calibration error vs. number of labels (k) |
| `exstress_fewshot_calib.csv` | Exercise-Stress few-shot calibration (boundary case) |
| `nurse_dca_raw_vs_recal.csv` | Net benefit, raw vs. recalibrated, across thresholds |
| `nurse_decision_curve.csv` | Full net-benefit curve (figure input) |
| `nurse_reliability.csv` | Reliability-diagram bins (figure input) |
| `nurse_loso_pred.csv` / `nurse_loso_recal.csv` | Raw and globally recalibrated Nurse predictions |

---

## Figures

| File | Content |
|---|---|
| `figures/Figure01_mimari.png` | Pipeline / architecture overview |
| `figures/figure02_discrimination.png` | Discrimination (AUROC) across the three datasets |
| `figures/figure03_label_efficiency.png` | Deep / SSL vs. baseline at increasing label fractions |
| `figures/figure04_calibration_problem.png` | Reliability diagrams and calibration slope under subject shift |
| `figures/figure05_fewshot_calibration.png` | Few-shot per-subject recalibration recovering calibration |
| `figures/figure06_decision_curve.png` | Decision-curve analysis: net benefit before vs. after recalibration |

---

## Summary of findings (LOSO)

- **WESAD** (controlled): near-ceiling discrimination (per-subject AUROC ≈ 0.99). Treated as a benchmark ceiling artifact, not a contribution claim.
- **Nurse** (real-world): modest discrimination (per-subject AUROC ≈ 0.60, pooled 0.63) with severe miscalibration (calibration slope ≈ 0.45). Few-shot per-subject recalibration cuts expected calibration error (k = 20: ≈ 0.09; k = 40: ≈ 0.07) and raises the share of decision-useful thresholds from ≈ 58 % to ≈ 79 %.
- **Exercise-Stress**: replicates the miscalibration (slope ≈ 0.22) but has too few windows per subject (≈ 20) for the few-shot remedy — a data-availability boundary condition reported honestly.
- A compact 1D-CNN, with or without self-supervised pre-training, never outperforms the calibrated feature + gradient-boosting baseline (DeLong p < 1e-6). Reported as an honest negative result.

---

## Citation

If you use this code, please cite the paper (citation to be updated upon acceptance):

```bibtex
@unpublished{akkaya2026calibration,
  title  = {Calibration, not architecture, limits cross-subject wearable stress detection: a multi-dataset, decision-focused evaluation with label-efficient recalibration},
  author = {Akkaya, Ahmet},
  year   = {2026},
  note   = {Manuscript under review}
}
```

Please also cite the original dataset papers (WESAD, Nurse Stress, Exercise-Stress) listed above.

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

---

## Contact

Ahmet Akkaya — Department of Computer Technologies, Bandırma Onyedi Eylül University, Gönen, Balıkesir, Türkiye
ORCID: [0000-0003-4836-2310](https://orcid.org/0000-0003-4836-2310) · aakkaya@bandirma.edu.tr
