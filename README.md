# Agreement-Guided Fusion (AGF) of attribution maps for brain tumour MRI

Code for the BSc thesis *Beyond Single Explanations: Agreement-Guided Fusion of Attribution Methods
for Explainable Brain Tumour MRI Classification* (Tamanna Jarin Tima, supervised by Dr. Imran
Mahmud, Department of Software Engineering, Daffodil International University, 2026) and the
accompanying MethodsX method article.

AGF is a training-free method that fuses Grad-CAM++, Integrated Gradients and SmoothGrad maps into
one explanation. Each map is weighted by a softmax (temperature τ) of its top-10% Dice agreement with
the other maps; when all maps agree equally, AGF reduces to Equal Fusion. The method comes with a
validation-locked evaluation protocol for faithfulness (insertion and deletion), robustness
(max-sensitivity) and localisation against expert tumour masks, and with a direct test of whether
agreement indicates faithfulness.

## Repository contents

```
notebooks/
  01_classification_benchmarking.ipynb      train and evaluate VGG16-BN, DenseNet121, ResNet50
  02_localisation_mask_matching.ipynb       match figshare tumour masks to the test images
  03_agf_main_evaluation.ipynb              attribution maps, AGF, faithfulness, robustness, ablations
  04_agf_tau_selection_and_analyses.ipynb   tau selection on validation, RQ analyses, localisation, oracle
  05_figures.ipynb                          qualitative grid and insertion/deletion curves
  extras/
    pairwise_dice.ipynb                     mean pairwise Dice between methods (from notebook 04 output)
    mcnemar_holm_correction.ipynb           Holm correction of the McNemar tests (from notebook 01 output)
src/
  xai_attribution.py                        the attribution and AGF module (read-only copy, see below)
```

The notebooks are the versions that produced the reported results and keep their printed outputs,
except `02_localisation_mask_matching`, which is stored without outputs; its outputs
(`test_masks_224.npz`, `test_mask_meta.json`) are read by notebook 04.

## Run order and inputs

All notebooks were run on Kaggle with a GPU (Tesla T4). Each one reads the outputs of the earlier
notebooks as Kaggle *Inputs* (Add Input → Notebook output).

| Step | Notebook | Kaggle inputs | Main outputs |
|---|---|---|---|
| 1 | `01_classification_benchmarking` | Brain Tumor MRI Dataset | `models/*_best.pth` checkpoints, metrics, McNemar tables |
| 2 | `02_localisation_mask_matching` | Brain Tumor MRI Dataset; figshare brain tumour dataset (`.mat` files) | `test_masks_224.npz`, `test_mask_meta.json` |
| 3 | `03_agf_main_evaluation` | Brain Tumor MRI Dataset; output of step 1 | per-image results, statistics, ablations, AOPC curves |
| 4 | `04_agf_tau_selection_and_analyses` | Brain Tumor MRI Dataset; outputs of steps 1 and 2; output of step 3 (optional, for the reproduction check) | `tau_selection.json`, RQ1–RQ3 tables, localisation, oracle, `revision_outputs.zip` |
| 5 | `05_figures` | Brain Tumor MRI Dataset; outputs of steps 1 and 3; output of step 4 (optional, supplies τ\*) | `qualitative_grid.pdf`, `aopc_curves_<backbone>.pdf` |

Set the accelerator to GPU and use **Run All**. The notebooks use Kaggle paths (`/kaggle/input/...`,
`/kaggle/working/...`); to run elsewhere, change the input and output directories in the
configuration cell at the top of each notebook.

Do not edit the configuration cell **R0** in notebook 04 after the first run: the temperature grid,
selection rule and settings are hashed into `tau_selection.json` before selection, and the test
evaluation stops if the selected values do not match this record.

## Data

Neither dataset is redistributed here.

- **Brain Tumor MRI Dataset, Version 2** (M. Nickparvar), 7,200 images in four classes:
  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- **figshare brain tumour dataset** (J. Cheng et al., 2015), used only for expert tumour masks and
  patient identifiers: https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset
  (original: https://doi.org/10.6084/m9.figshare.1512427)

## Trained weights

The three trained classifiers are available at: https://doi.org/10.5281/zenodo.24594903

## Environment

Notebooks 01–04 were run with Python 3.12.13, PyTorch 2.10.0 (CUDA 12.8), timm 1.0.26, NumPy 2.0.2,
SciPy 1.16.3, pandas 2.3.3, statsmodels 0.14.6 and OpenCV 4.13.0. Notebook 05 was run later on a newer
Kaggle image (Python 3.13, PyTorch 2.11, timm 1.0.29); it only redraws figures. See
`requirements.txt`.

## Reproducibility safeguards

- All random number generators are seeded (global seed 42; separate fixed seeds for each subset),
  with deterministic cuDNN settings and FP32 attribution.
- The explained class is the clean prediction, fixed for all methods and analyses.
- A prediction-parity check compares the attribution pipeline with the classifiers on all 1,600
  test images and stops below 98% agreement (observed 99.81–99.94%).
- An agreement-kernel self-check on a synthetic control runs before any attribution.
- Target layers are verified (expected module, 7×7 output, non-negative activations).
- The temperature τ\* is selected on validation images only and locked before testing.
- A repeated run reproduced the insertion scores of the three base methods, Equal Fusion and AGF (τ = 1) to within 1.1 × 10⁻¹⁶.

## About `src/xai_attribution.py`

The notebooks write this module themselves (a `%%writefile` cell) and import it from
`/kaggle/working`. The file in `src/` is an identical copy, kept so the method can be read and reused
without opening the notebooks.

## Citation

If you use this code, please cite the thesis or the MethodsX article (see `CITATION.cff`).

## License

Code: MIT License (see `LICENSE`). The datasets keep their own licences.
