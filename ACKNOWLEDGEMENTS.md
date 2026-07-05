# Acknowledgements

This project builds on several open-source works. We thank their authors.

- **DiffMorpher** — *DiffMorpher: Unleashing the Capability of Diffusion Models
  for Image Morphing* (CVPR 2024), Zhang et al.
  <https://github.com/Kevin-thu/DiffMorpher>
  All of `stage1_morph/` (the diffusion morphing pipeline: `model.py`, `utils/`,
  `morph_pair.py`) is derived from this repository and is distributed under the
  **S-Lab License 1.0** (non-commercial research use). See `stage1_morph/LICENSE`.

- **RoMa** — *RoMa: Robust Dense Feature Matching* (CVPR 2024), Edstedt et al.
  <https://github.com/Parskatt/RoMa>
  Used as the dense correspondence estimator in `stage2_register/compose_flow.py`
  (installed as an external dependency, the `romatch` package).

- **Stable Diffusion 2.1** (Stability AI) is the diffusion backbone loaded by the
  morphing stage via `diffusers`.

Datasets (LEVIR-CD, WHU-CD, DSIFN-CD) belong to their respective authors; please
cite them and follow their licenses (see the README).
