# ct-xai-qc

**A quality-control protocol for saliency-based explanations in CT: faithfulness, stability under acquisition shift, and sanity checks.**

[![tests](https://github.com/avossirini9404/ct-xai-qc/actions/workflows/ci.yml/badge.svg)](https://github.com/avossirini9404/ct-xai-qc/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

---

## Motivation

Before a CT scanner is used on patients, someone measures it. Not once: a quality-control
programme fixes what is measured, how often, against which tolerance, and what happens when the
tolerance is exceeded. The measurements include negative controls, because an instrument that
reports the same reading with the sample removed is not measuring the sample. None of this asks
whether the scanner is *useful*. It asks whether its output is reproducible under the conditions
the clinic actually presents.

Saliency maps are increasingly offered as the reason a deep-learning model made a decision about a
CT image, and they are almost never held to that standard. This repository treats a saliency map as
a measuring instrument and asks the three questions a quality-control programme would ask of one:

1. **Is it faithful?** Does removing the pixels it ranks highest actually change the model's
   output, or does it point somewhere decorative? (Deletion and insertion curves.)
2. **Is it stable?** Dose, reconstruction kernel, calibration and contrast phase vary between
   examinations and between institutions. When they vary and *the prediction does not change*, does
   the explanation stay the same?
3. **Does it pass its negative control?** If the model's weights are destroyed layer by layer and
   the map barely moves, the map is a property of the image and the architecture, not of anything
   the model learned.

The second question is the one this repository is built around, and the reason it uses physics
rather than additive Gaussian noise. A model tested only against Gaussian noise is being asked a
question no scanner ever asks.

## Main result

<!-- ![Explanation agreement against acquisition change](results/figures/stability_vs_dose.png) -->

*Produced by `make report` from your own run, with `configs/default.yaml`.*

Read the two vertical axes together. The left-hand curves are the model's output: balanced accuracy
and the fraction of predictions that agree with the reference acquisition. The right-hand curves are
the agreement of the saliency map with its own reference map, measured **only on the images whose
predicted class did not change**. Wherever the left-hand curves stay flat and the right-hand ones
fall, the prediction survived the acquisition change and the explanation offered for it did not —
and only the second one is what a reader is shown.


A worked example on the synthetic phantom dataset that ships with the repository, which needs no
download and runs in under a minute (`make smoke`), is committed at
[`results/figures/example_synthetic_hu_shift.png`](results/figures/example_synthetic_hu_shift.png):
at a 50 HU calibration offset every prediction is unchanged and balanced accuracy is 1.00, while the
overlap of the top 10 % most salient pixels with the reference map has already fallen from 0.94 to
0.68. At 100 HU, 98 % of predictions still agree and that overlap is 0.17. The full table for that
run is [`results/tables/example_synthetic_summary.md`](results/tables/example_synthetic_summary.md).
The synthetic phantom is a sanity check on the protocol, not a finding about CT.

Grad-CAM passes its negative control on that run: agreement with the trained model's map falls to
0.39 after the classifier is re-initialised and to roughly zero once the convolutional blocks are,
which is the behaviour a usable explanation should show. It also beats random pixel ranking on both
faithfulness curves, though not by a wide margin.

## What's here

| Path | What it does |
| --- | --- |
| `src/ctxaiqc/perturb/` | Acquisition simulator. Forward projection, Poisson photon statistics, filtered back-projection, reconstruction kernel, HU calibration offset, contrast gain, slice-thickness surrogate. |
| `src/ctxaiqc/explain/` | Grad-CAM, Integrated Gradients and occlusion, implemented from the original papers rather than imported, so every step the protocol measures is visible here. |
| `src/ctxaiqc/metrics/` | Performance, expected calibration error and Monte-Carlo dropout, deletion/insertion faithfulness, three measures of saliency agreement, cascading-randomisation sanity check. |
| `src/ctxaiqc/models/` | A small CNN (default) and an optional ResNet-18. |
| `src/ctxaiqc/run_experiment.py` | The protocol: four CSV tables under `results/tables/`. |
| `results/` | Tables and figures written by `make experiment` and `make report`, plus the committed synthetic-phantom example. |
| `notebooks/` | Four notebooks that walk through the argument end to end. |
| `configs/` | `default.yaml` (OrganAMNIST, 64 px), `nodule.yaml` (NoduleMNIST3D), `smoke.yaml` (synthetic, offline, what CI runs). |

### Data

[MedMNIST v2](https://medmnist.com/): **OrganAMNIST** (axial abdominal CT, 11 organ classes, derived
from the LiTS benchmark) by default, and **NoduleMNIST3D** (LIDC-IDRI, benign vs malignant) as a
binary alternative, from which the central axial slice is taken so that one 2-D architecture serves
both. Both download automatically. A synthetic phantom dataset is generated in-process for the
tests, so the test suite needs no network access.

### One design decision worth stating

The benchmark image is treated as **the object, not as an image that has already been through a
scanner**. Every image the pipeline produces — the training set, the reference acquisition, and each
perturbed acquisition — is the output of one simulated acquisition, and a perturbation family moves
exactly one setting of it. Training on the raw arrays instead would put the reference level of every
family out of distribution, and the measured degradation would be reporting the simulator rather
than the perturbation.

## Reproduce it

```
python -m pip install -e ".[data,dev]"
python -m ctxaiqc.train          --config configs/default.yaml
python -m ctxaiqc.run_experiment --config configs/default.yaml
python -m ctxaiqc.report         --config configs/default.yaml
```

**What the default configuration actually runs.** `configs/default.yaml` trains on a stratified
subset of 4 000 OrganAMNIST images at 64 px for 10 epochs, and evaluates the perturbation protocol
on 400 test images, of which 64 are used for the saliency comparison. That is a deliberate choice:
the acquisition simulator runs on CPU and is the bulk of the wall-clock time, and the protocol
compares a model against itself under acquisition change rather than against an accuracy threshold,
so a smaller training set costs accuracy without invalidating the comparison. Remove the `subset`
block to train on the full dataset; the numbers reported here are from the subset.

Timings measured on [your CPU], no GPU. The first run additionally spends [X] minutes simulating the
reference acquisition of every image it will use; the result is cached under `data/cache/`. A GPU
shortens training but not the simulator.

`make smoke` runs the whole pipeline on synthetic phantoms in under a minute and needs no download.
`make test` runs 33 unit and smoke tests, offline.

Every run is determined by the seed in the configuration file, which is recorded inside the
checkpoint together with the configuration that produced it.


## Methods

**Acquisition simulation.** The image is mapped to Hounsfield units under a stated convention
(0 → −1000 HU, 1 → +1000 HU), converted to linear attenuation coefficients through the definition of
the HU scale, and projected with the Radon transform. Photon counts follow Beer–Lambert,
`N = I₀ exp(−p)`, and are drawn from a Poisson distribution, which is the noise model for photon
counting in the quantum-limited regime; lowering the dose lowers `I₀`. Reconstruction is filtered
back-projection with a selectable filter. Calibration offsets are applied after reconstruction and
contrast changes before projection, because a scanner calibration offset is added to reconstructed
HU values while a beam-hardening-driven contrast change alters the projections themselves.

**Explanations.** Grad-CAM (Selvaraju et al., 2017), Integrated Gradients (Sundararajan et al.,
2017), occlusion (Zeiler & Fergus, 2014).

**Metrics.** Deletion and insertion curves (Petsiuk et al., 2018), each compared against a
random-ranking control. Expected calibration error (Guo et al., 2017) and Monte-Carlo dropout (Gal &
Ghahramani, 2016). Saliency agreement as Spearman rank correlation, structural similarity, and
Jaccard overlap of the top 10 % of pixels — three views because each fails differently. Cascading
randomisation (Adebayo et al., 2018) as the negative control.

**Protocol.** Saliency maps are compared **only on images whose predicted class is unchanged**, so
that a loss of agreement cannot be attributed to the model having changed its mind.

<details>
<summary>References</summary>

- Adebayo, J. et al. Sanity checks for saliency maps. *NeurIPS*, 2018.
- Gal, Y. & Ghahramani, Z. Dropout as a Bayesian approximation. *ICML*, 2016.
- Guo, C. et al. On calibration of modern neural networks. *ICML*, 2017.
- Lekadir, K. et al. FUTURE-AI: international consensus guideline for trustworthy and deployable artificial intelligence in healthcare. *BMJ*, 2025.
- Petsiuk, V. et al. RISE: randomized input sampling for explanation. *BMVC*, 2018.
- Selvaraju, R. R. et al. Grad-CAM. *ICCV*, 2017.
- Sundararajan, M. et al. Axiomatic attribution for deep networks. *ICML*, 2017.
- Yang, J. et al. MedMNIST v2. *Scientific Data*, 2023.
- Zeiler, M. D. & Fergus, R. Visualizing and understanding convolutional networks. *ECCV*, 2014.

</details>

## Scope and limitations

This is a methods demonstration on a public benchmark, and the following are real limits on what it
can support, not disclaimers:

- **The data are 2-D, low-resolution benchmark images, not clinical CT.** MedMNIST distributes 8-bit
  crops whose original HU window is not recoverable, so the mapping to Hounsfield units is a stated
  convention rather than a recovered calibration, and every quantity expressed in HU inherits that.
  For NoduleMNIST3D, taking the central slice discards information the 3-D task provides.
- **The acquisition model is an approximation, not a CT simulator.** Parallel-beam geometry,
  monoenergetic photons, no beam hardening, no scatter, no detector response, no bowtie filter, no
  tube-current modulation, no iterative or deep reconstruction. It reproduces the direction and
  rough magnitude of the effects it models; it does not reproduce a specific scanner. The
  slice-thickness family is an in-plane blur standing in for an axis the data do not have.
- **There is no real multi-centre external validation.** Simulated acquisition shift and genuine
  inter-institutional heterogeneity — different vendors, protocols, populations and reconstruction
  software — are not the same thing, and agreement under the first says little about the second.
- **One small architecture, one training run per configuration.** No architecture sweep, no seed
  variance study, no confidence intervals on the agreement curves beyond the reported standard
  deviations.
- **Post-hoc saliency is a weak form of explanation to begin with.** The natural next step is
  generative counterfactuals — asking what minimal, realistic change to the image would flip the
  prediction — rather than measuring which pixels a gradient happens to weight. That work is
  declared on the `roadmap` branch and is not implemented here; a small 2-D diffusion model for
  counterfactual generation is the intended direction.

Stating these precisely is part of the exercise. A quality-control programme that does not record
what it did *not* measure is not a quality-control programme.

## Citation

See [`CITATION.cff`](CITATION.cff). Released under the MIT licence.

---

Arianna Ruiz Vossirini · [ORCID 0009-0000-2557-9061](https://orcid.org/0009-0000-2557-9061)
