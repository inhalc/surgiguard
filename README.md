<div align="center">

# SurgiGuard

### Reliable Intraoperative Video Segmentation

[![KDD 2026](https://img.shields.io/badge/KDD-2026-25201e)](https://doi.org/10.1145/3770855.3818851)
![Python](https://img.shields.io/badge/python-3.10+-c74d39)
![License](https://img.shields.io/badge/license-Apache--2.0-726862)

**Project Lead & First Author — Jiutao Zhou**

</div>

![Synthetic SurgiGuard demonstration](assets/hero.png)

SurgiGuard is a model-agnostic temporal reliability layer for video segmentation. It aligns recent
probability maps to the current frame, builds a confidence-weighted history reference, and releases
the temporal anchor where current evidence indicates genuine change. The public demo is synthetic;
it uses no clinical imagery or hospital source code.

## Implemented capabilities

- **Motion-aware history:** dense optical flow warps each stored probability map into the current
  frame and discounts inconsistent or out-of-bounds correspondence.
- **Stability–response control:** a per-pixel gate combines the aligned reference with the current
  model probability instead of applying uniform smoothing.
- **Stream-safe monitoring:** independent per-stream state, explicit reset, time-order validation,
  flicker/drift metrics, and an area-growth trend alert.

![Implemented architecture](assets/architecture.png)

The included synthetic sequence contains target motion, a short prediction artifact, recovery, and
real area growth. Its four panels are produced by the same controller used by the Python API.

![Synthetic qualitative sequence](assets/qualitative_results.png)

## Try the complete example

```bash
python -m pip install -e ".[app,dev]"
python -m pytest -q
streamlit run app/streamlit_app.py
```

To connect an authorized segmenter, provide a callable that receives a BGR frame and returns a
two-dimensional probability map in `[0, 1]`:

```bash
python scripts/run_video.py procedure.mp4 --predictor my_model:predict_probability
```

The single-process demonstration API is available with:

```bash
uvicorn surgiguard.service.api:app --reload
```

Core entry points are `temporal/controller.py` for stateful inference, `pipeline.py` for monitoring,
and `service/api.py` for isolated stream sessions. See [the design note](docs/design.md) for the
technical choices and boundaries.

## Publication results

The associated KDD 2026 paper reports the following comparison on 300 intraoperative sequences.
These are paper results—not measurements from the synthetic public demo.

| Measure | nnU-Net | Proposed method |
|---|---:|---:|
| Dice | 0.62 ± 0.03 | **0.73 ± 0.02** |
| Centroid drift (px) | 5.8 ± 0.6 | **2.6 ± 0.4** |
| Flicker rate | 0.21 ± 0.03 | **0.09 ± 0.01** |
| Trend-prompt lead time (s) | 1.0 ± 0.3 | **2.2 ± 0.4** |

The paper also evaluates three public surgical benchmarks; consult the publication for protocols,
baselines, and full tables.

**Jiutao Zhou**, Xiaoyang Li, Yuhao Zhang, Xiaoqian Peng, Weiguang Qu, Peirong Ma, and Yanhui Gu.
“Controlling Prediction Dynamics for Reliable Intraoperative Segmentation.” *KDD 2026*.
[DOI and paper](https://doi.org/10.1145/3770855.3818851) · [Citation metadata](CITATION.cff)

## Public release boundary

This repository is a clean, data-free implementation of the published system design. Hospital
production source, clinical data, annotations, private configuration, and model weights are not
included. The software is a research artifact, not an autonomous diagnostic or treatment system.

Released under the [Apache License 2.0](LICENSE).
