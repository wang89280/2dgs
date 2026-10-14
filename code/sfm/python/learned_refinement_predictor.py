"""Pairwise inference through the original, pinned Keypt2Subpx ALIKED network.

This optional module does not select tracks, change pixels in-place, run BA, or
evaluate holdout data. The bundled official weights were trained with ALIKED-N16;
using this project's N32 descriptors is an explicitly unvalidated transfer.

Patch-provider contract for each side:
  keypoints_native: (N, 2), the project's full-image coordinates including +0.5;
  descriptors: (N, 128), the original L2-normalized ALIKED descriptors;
  rawpatch/scorepatch: (N, 1, 11, 11), float32 in [0, 1];
  offset_scale_native: (N, 2), normally ((tile_w-1)/tile_w,(tile_h-1)/tile_h).

The original GlueFactory commit restores normalized ALIKED points using (w,h),
whereas this project uses (w-1,h-1) and adds 0.5. Patches must be extracted at
official_xy=(tile_native_xy-0.5)*(w,h)/(w-1,h-1), with the original integer crop,
zero padding and clamp in third_party/keypt2subpx/{model,utils}.py. The scale above
converts predicted official-coordinate offsets back to native coordinates.
"""
from __future__ import annotations

import builtins
import hashlib
import importlib
import json
from pathlib import Path
import types

import numpy as np
import torch


VENDOR = Path(__file__).resolve().parents[1] / 'third_party/keypt2subpx'
LIMITATION = ('Official Keypt2Subpx ALIKED-N16 pretrained network applied to '
              'ALIKED-N32 features; descriptor and score-map distribution '
              'equivalence and reconstruction accuracy gains are not established.')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_official_module(path: Path, name: str, dependencies=None):
    """Keep the upstream source unmodified and isolate its plain `utils` import."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    original_import = builtins.__import__
    dependencies = dependencies or {}

    def local_import(module_name, globals=None, locals=None, fromlist=(), level=0):
        if not level and module_name in dependencies:
            return dependencies[module_name]
        return original_import(module_name, globals, locals, fromlist, level)

    module.__dict__['__builtins__'] = dict(vars(builtins), __import__=local_import)
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), module.__dict__)
    return module


def _array(value, name, shape, *, unit_interval=False):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(f'{name}: expected shape {shape}, received {result.shape}')
    if not np.isfinite(result).all():
        raise ValueError(f'{name}: nonfinite input')
    if unit_interval and result.size and (result.min() < -1e-6 or result.max() > 1+1e-6):
        raise ValueError(f'{name}: expected raw values in [0, 1], without normalization')
    return np.ascontiguousarray(result)


class LearnedRefinementPredictor:
    """Load the pinned original AttnTuner with the official ALIKED checkpoint."""

    def __init__(self, device='cuda', batch_size=2048, vendor_dir=None):
        self.vendor = Path(vendor_dir) if vendor_dir is not None else VENDOR
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; explicitly choose device="cpu"')
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Supported devices are cpu and cuda')
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError('batch_size must be positive')
        manifest_path = self.vendor / 'SOURCE_MANIFEST.json'
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        records = {r['path']: r for r in self.manifest['files']}
        required = ['model.py', 'utils.py', 'pretrained/k2s_aliked_pretrained.pth']
        for name in required:
            path = self.vendor / name
            if name not in records or _sha256(path) != records[name]['sha256']:
                raise RuntimeError('Official source or weight hash mismatch: ' + str(path))
        self.official_utils = _load_official_module(self.vendor/'utils.py', '_seed_k2s_utils')
        self.official_model = _load_official_module(
            self.vendor/'model.py', '_seed_k2s_model', {'utils': self.official_utils})
        self.network = self.official_model.AttnTuner(128, use_score=True)
        # The verified upstream checkpoint contains one NumPy float64 metadata
        # scalar. NumPy 2 renamed its implementation module; map only this known
        # numeric global, retaining weights_only rather than enabling pickle code.
        scalar = importlib.import_module('numpy.core.multiarray').scalar
        safe_numeric_globals = [(scalar, 'numpy.core.multiarray.scalar'),
                                np.dtype, type(np.dtype(np.float64))]
        with torch.serialization.safe_globals(safe_numeric_globals):
            checkpoint = torch.load(self.vendor/'pretrained/k2s_aliked_pretrained.pth',
                                    map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
            raise RuntimeError('Unexpected official checkpoint structure')
        self.network.load_state_dict(checkpoint['model'], strict=True)
        self.network.to(self.device).eval()
        self.metadata = {
            'method': 'official_Keypt2Subpx_ALIKED_pairwise_inference',
            'upstream_repository': self.manifest['repository'],
            'upstream_commit': self.manifest['commit'],
            'license': self.manifest['license'],
            'weights_sha256': records[required[-1]]['sha256'],
            'torch_version': torch.__version__,
            'device': str(self.device),
            'batch_size': self.batch_size,
            'source_detector': 'ALIKED-N16',
            'target_detector': 'ALIKED-N32',
            'limitation': LIMITATION,
            'patch_contract': 'official integer 11x11 RGB-to-gray / raw sigmoid score patches',
            'patch_coordinate_transform': '(tile_native_xy-0.5)*(w,h)/(w-1,h-1)',
            'offset_coordinate_transform': 'offset_official*(w-1,h-1)/(w,h)',
            'multiview_aggregation_performed': False,
            'ba_performed': False,
            'reads_test_files': False,
        }

    def predict_patches(self, keypoints0_native, keypoints1_native,
                        descriptors0, descriptors1, rawpatch0, rawpatch1,
                        scorepatch0, scorepatch1, *,
                        offset_scale0_native, offset_scale1_native):
        """Return pairwise candidate pixels; no rejection or hidden BA is applied.

        Ordering is the matched-pair ordering on both sides. Callers retain their
        own image/feature IDs and must resolve repeated-feature predictions.
        Provided patches must come from the originating detector tiles. Cached
        sparse scalar scores are not a substitute for 11x11 score-map patches.
        """
        if isinstance(keypoints0_native, torch.Tensor):
            n = int(keypoints0_native.shape[0])
        else:
            n = len(keypoints0_native)
        kp0 = _array(keypoints0_native, 'keypoints0_native', (n, 2))
        kp1 = _array(keypoints1_native, 'keypoints1_native', (n, 2))
        d0 = _array(descriptors0, 'descriptors0', (n, 128))
        d1 = _array(descriptors1, 'descriptors1', (n, 128))
        for name, desc in [('descriptors0', d0), ('descriptors1', d1)]:
            if n and np.max(np.abs(np.linalg.norm(desc, axis=1)-1)) > 0.01:
                raise ValueError(name + ': expected original L2-normalized descriptors')
        patches = [_array(x, name, (n, 1, 11, 11), unit_interval=True) for name, x in [
            ('rawpatch0', rawpatch0), ('rawpatch1', rawpatch1),
            ('scorepatch0', scorepatch0), ('scorepatch1', scorepatch1)]]
        scales = []
        for name, value in [('offset_scale0_native', offset_scale0_native),
                            ('offset_scale1_native', offset_scale1_native)]:
            scale = _array(value, name, (n, 2))
            if n and ((scale <= 0).any() or (scale > 1).any()):
                raise ValueError(name + ': expected (tile_size-1)/tile_size in (0, 1]')
            scales.append(scale)
        offsets = [np.empty((n, 2), np.float32), np.empty((n, 2), np.float32)]
        with torch.inference_mode():
            for begin in range(0, n, self.batch_size):
                end = min(n, begin+self.batch_size)
                # The original implementation averages descriptors without an
                # additional normalization; changing that would change its model.
                mean = torch.from_numpy((d0[begin:end]+d1[begin:end])*0.5).to(self.device)[None]
                for side in (0, 1):
                    raw = torch.from_numpy(patches[side][begin:end]).to(self.device)[None]
                    score = torch.from_numpy(patches[side+2][begin:end]).to(self.device)[None]
                    prediction = 2.5*self.network(raw, score, mean)
                    offsets[side][begin:end] = prediction[0].cpu().numpy()
        if not all(np.isfinite(x).all() for x in offsets):
            raise RuntimeError('Official refiner returned nonfinite offsets')
        native = [offsets[i]*scales[i] for i in (0, 1)]
        return {
            'refined0_native': kp0+native[0],
            'refined1_native': kp1+native[1],
            'offset0_native': native[0],
            'offset1_native': native[1],
            'offset0_official': offsets[0],
            'offset1_official': offsets[1],
            'metadata': dict(self.metadata, pairs=n),
        }


def load_predictor(device='cuda', batch_size=2048):
    return LearnedRefinementPredictor(device=device, batch_size=batch_size)


Predictor = LearnedRefinementPredictor
