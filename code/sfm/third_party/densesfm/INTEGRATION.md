# DenseSfM refinement integration

Upstream: https://github.com/IceTea-CV/DenseSfM-Refine

Paper: *Dense-SfM: Structure from Motion with Dense Consistent Matching*,
CVPR 2025, https://arxiv.org/abs/2501.14277.

The pinned commit, original checkpoint SHA-256 and tensors-only checkpoint
SHA-256 are recorded in `provenance.json`. The released MultiviewMatcher files
are retained in `densesfm_network/`. This is attributed reuse of a pretrained
network, not a claim to have independently implemented or trained that network.

Local changes are limited to two ROI imports and lazy loading of unused ResNet
code. `python/densesfm_roi.py` implements the endpoint crop semantics of the
upstream RoIAlign extension in PyTorch. It is numerically tested against an
independent bilinear reference including out-of-bounds coordinates. The adapter
is tested against the retained upstream forward method, including unequal image
scales, movable/frozen anchors, and batch-size invariance.

ALIKED-N32 and LightGlue are still the feature extraction/matching frontend.
Some upstream decoder helper filenames contain `roma`; these are components of
the released refinement network. No RoMa/RoMa-v2 matcher, frontend model or
additional dense correspondences are loaded.

This integration uses one refinement pass, track groups of at most 16 views,
1200-pixel RGB images, the 35-pixel crop and the original 15/7-pixel search windows.
Long tracks retain one anchor, updated in their first segment and frozen in
subsequent segments. There is no fine-tuning, checkpoint selection, confidence
threshold search or geometric reprojection used as a measurement target.

Geometric optimization is the existing native fixed-intrinsic robust BA, with
three calls for both control and candidate. The new native target changes only
the accepted input displacement bound (64 pixels), not the objective, gauge,
solver, filtering or BA budget. Python additionally checks the actual network
search window and image bounds. If a predicted observation is invalid, its
entire track keeps the original measurements. No observation/track is deleted.

This is not the full DenseSfM system: it does not use their customized COLMAP
refiner, GS track extension, alternating refinement schedule or track filtering.
Their repo's top-level checkout had no explicit license file; preserve upstream
attribution and do not infer a license from related projects. Existing upstream
comments and notices are retained.

Runtime: CUDA PyTorch/torchvision, numpy, Pillow, einops, kornia, loguru,
OmegaConf. The benchmark records hashes and uses the downloaded checkpoint with
strict tensor matching; missing/mismatched parameters cause an error.

The original Lightning checkpoint contains metadata. `tools/prepare_densesfm_weights.py`
performs a safe allowlisted load with an inert checkpoint-metadata placeholder,
then extracts only tensor parameters. Runtime loads `refinement_weights.pt` with
`weights_only=True`, without Lightning or executable checkpoint objects.
