2DGS foreground stability repair (2026-10-14)

This isolated Python runtime is derived from the package's frozen `code/2dgs`.
The original Python files, renderer, trained models, ALIKED/LightGlue frontend,
and SfM reconstructions are not modified. It uses the existing WSL
`surfel_splatting` environment and its unchanged compiled CUDA extensions.
The original license is retained in LICENSE.md.

Changes enabled by `--use_masks --foreground_stability`:

- Original full-image L1/DSSIM photometric objective is preserved.
- Supervise opacity in confident foreground/background with weights 0.01/0.1;
  exclude a 3-pixel silhouette band. Only training masks are loaded after splitting.
- Normalize both normal vectors and constrain only reliable foreground. Normal
  weight ramps from 0 to 0.05 over iterations 15,000–20,000.
- Preserve existing screen-radius statistics through clone/split until pruning;
  only newly appended Gaussians receive zero statistics. Reset after consumption.
- Use a conservative screen-radius bound of 25% image width (400 px at 1600 px).
  The historical 20 px test had been ineffective due to the reset bug; enabling
  it unchanged would introduce much more aggressive pruning.
- Continue safety pruning every 1,000 steps after densification, except at the
  final reporting step: opacity <0.005, scale >0.1*extent, or radius >screen bound.
- Store Python/NumPy/Torch/CUDA RNG state and remaining view stack in checkpoints.

Runner: `tools/run_foreground_fix_2dgs.py` in the package root. It copies this
runtime to WSL and freezes hashes/options before three 30k runs (ON, OFF, COLMAP).
It refuses accidental restarts; inspected interruptions may use `--resume`.
All three runs use 2-1, seed 0, 70 training/10 evaluation views and identical
repair options. Final 30k models are evaluated without mask compositing.

Masks are derived solely from existing black-background training RGB, using
the largest connected component above 2/255 intensity with filled holes.
This is a dataset-specific engineering repair, not a new SfM algorithm or a
claim of independent geometric accuracy. Test images are not used to tune
weights or select a checkpoint. Per-module contribution requires a separate
ablation; this batch tests the joint repair package.
