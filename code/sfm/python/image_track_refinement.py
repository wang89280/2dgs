"""Image-only, identity-preserving subpixel refinement of existing SfM tracks.

This is an engineering variant of classical local image alignment, not PixSfM.
No projection, pose, 3-D point, calibration parameter, or evaluation observation
is used in deciding a displacement.  The input observations are never edited.

Coordinates in the native model use COLMAP pixel centers.  OpenCV LK and remap
use integer pixel centers, so all image operations subtract 0.5, exactly once.
The confidence/sigma outputs are bounded image-evidence heuristics, NOT a
statistically calibrated uncertainty estimate.  Uniform-weight BA is therefore
the appropriate first controlled comparison.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from model_io import OBS


@dataclass(frozen=True)
class Policy:
    # Fixed engineering assumptions, selected without benchmark/test metrics.
    camera_sequence_gap: int = 3
    neighbors_each_direction: int = 2
    lk_window: int = 15
    lk_levels: int = 2
    lk_iterations: int = 30
    lk_epsilon: float = 0.005
    min_eigenvalue: float = 0.0001
    patch_size: int = 11
    minimum_patch_std: float = 5.0  # 8-bit gray intensity units
    minimum_zncc: float = 0.75
    minimum_zncc_gain: float = 0.002
    maximum_antisymmetry_error: float = 0.8
    maximum_cycle_error: float = 0.5
    maximum_pair_displacement: float = 4.0
    maximum_observation_displacement: float = 2.0
    graph_regularization: float = 0.15
    graph_huber: float = 0.5
    graph_irls_iterations: int = 3
    minimum_edge_weight: float = 0.15
    sigma_floor: float = 0.6
    image_cache_size: int = 6
    cv_threads: int = 4


POLICY = Policy()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _image_paths(sfm: Path, images: Path, ids: np.ndarray) -> list[Path]:
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
    if not files:
        raise ValueError(f"No images in {images}")
    by_name = {p.name.casefold(): p for p in files}
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        by_stem[p.stem.casefold()].append(p)
    labels = {}
    xml = sfm / "cameras.xml"
    if xml.exists():
        # Only camera IDs and image labels are used; transforms are ignored.
        labels = {int(e.attrib["id"]): e.attrib.get("label", "")
                  for e in ET.parse(xml).getroot().findall(".//cameras/camera")}
    result = []
    for index in ids:
        label = labels.get(int(index), "")
        name = Path(label).name.casefold()
        if name in by_name:
            result.append(by_name[name])
        elif name and len(by_stem.get(Path(name).stem, [])) == 1:
            result.append(by_stem[Path(name).stem][0])
        elif not label and int(index) < len(files):
            result.append(files[int(index)])
        else:
            raise ValueError(f"Cannot resolve camera ID {index}, label={label!r}")
    if len(set(result)) != len(result):
        raise ValueError("Camera labels do not identify unique images")
    return result


class _Images:
    def __init__(self, paths: list[Path], policy: Policy):
        self.paths, self.policy = paths, policy
        self.cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.hashes: dict[str, str] = {}

    def get(self, camera: int) -> tuple[np.ndarray, np.ndarray]:
        camera = int(camera)
        if camera in self.cache:
            self.cache.move_to_end(camera)
            return self.cache[camera]
        path = self.paths[camera]
        # imdecode handles Windows Unicode paths reliably.
        raw = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if image is None or image.ndim != 2:
            raise ValueError(f"Cannot load grayscale image {path}")
        self.hashes[str(path.resolve())] = hashlib.sha256(raw.tobytes()).hexdigest()
        eigen = cv2.cornerMinEigenVal(image, blockSize=7, ksize=3)
        self.cache[camera] = image, eigen
        while len(self.cache) > self.policy.image_cache_size:
            self.cache.popitem(last=False)
        return self.cache[camera]


def _sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, np.float32)
    return cv2.remap(image, p[:, 0:1], p[:, 1:2], interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT).reshape(-1)


def _patches(image: np.ndarray, points: np.ndarray, size: int) -> np.ndarray:
    delta = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    dx, dy = np.meshgrid(delta, delta)
    p = np.asarray(points, np.float32)
    mx = p[:, 0, None] + dx.reshape(1, -1)
    my = p[:, 1, None] + dy.reshape(1, -1)
    # Remap has a 32767 row/column limit; callers batch <=2048 observations.
    patch = cv2.remap(image, mx, my, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
    return patch


def _patch_similarity(a: np.ndarray, b: np.ndarray, pa: np.ndarray,
                      pb: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    correlation = np.zeros(len(pa), np.float32)
    texture = np.zeros(len(pa), np.float32)
    for lo in range(0, len(pa), 2048):
        hi = min(lo + 2048, len(pa))
        aa, bb = _patches(a, pa[lo:hi], size), _patches(b, pb[lo:hi], size)
        aa -= aa.mean(axis=1, keepdims=True)
        bb -= bb.mean(axis=1, keepdims=True)
        sa = np.sqrt(np.mean(aa * aa, axis=1))
        sb = np.sqrt(np.mean(bb * bb, axis=1))
        correlation[lo:hi] = np.sum(aa * bb, axis=1) / np.maximum(
            np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1), 1e-6)
        texture[lo:hi] = np.minimum(sa, sb)
    return correlation, texture


def _inside(image: np.ndarray, points: np.ndarray, margin: float) -> np.ndarray:
    h, w = image.shape
    return np.isfinite(points).all(axis=1) & (points[:, 0] >= margin) & (points[:, 1] >= margin) & (points[:, 0] < w - 1 - margin) & (points[:, 1] < h - 1 - margin)


def _lk(a: np.ndarray, b: np.ndarray, pa: np.ndarray, pb: np.ndarray,
        policy: Policy) -> tuple[np.ndarray, np.ndarray]:
    flow, status, _ = cv2.calcOpticalFlowPyrLK(
        a, b, np.asarray(pa, np.float32).reshape(-1, 1, 2),
        np.asarray(pb, np.float32).reshape(-1, 1, 2).copy(),
        winSize=(policy.lk_window, policy.lk_window), maxLevel=policy.lk_levels,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                  policy.lk_iterations, policy.lk_epsilon),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=policy.min_eigenvalue)
    if flow is None:
        return pb.copy(), np.zeros(len(pa), bool)
    return flow.reshape(-1, 2), status.reshape(-1).astype(bool)


def _pair_evidence(a: np.ndarray, b: np.ndarray, ea: np.ndarray, eb: np.ndarray,
                   pa: np.ndarray, pb: np.ndarray, policy: Policy) -> dict[str, np.ndarray]:
    """Measure both directions from original observations, with real cycle tests."""
    count = len(pa)
    keys = ("zncc_before", "zncc_after", "patch_std", "texture_eigenvalue",
            "antisymmetry_error", "cycle_error", "pair_displacement", "weight")
    result = {name: np.zeros(count, np.float32) for name in keys}
    result["delta"] = np.zeros((count, 2), np.float32)
    result["accepted"] = np.zeros(count, bool)
    # A bitmask retains every failed gate, instead of one opaque rejection label.
    result["rejection_mask"] = np.ones(count, np.uint16)  # bit 0: border
    margin = max(policy.lk_window, policy.patch_size) // 2 + 2
    eligible = _inside(a, pa, margin) & _inside(b, pb, margin)
    which = np.flatnonzero(eligible)
    if not len(which):
        return result
    x, y = pa[which], pb[which]
    fab, sab = _lk(a, b, x, y, policy)
    fba, sba = _lk(b, a, y, x, policy)
    # Replace unsuccessful destinations before the cycle call to avoid NaNs.
    goodab = sab & np.isfinite(fab).all(axis=1)
    goodba = sba & np.isfinite(fba).all(axis=1)
    fab[~goodab] = y[~goodab]
    fba[~goodba] = x[~goodba]
    cab, scab = _lk(b, a, fab, x, policy)
    cba, scba = _lk(a, b, fba, y, policy)
    ab, ba = fab - y, fba - x
    delta = 0.5 * (ab - ba)
    antisym = np.linalg.norm(ab + ba, axis=1)
    cycle = np.maximum(np.linalg.norm(cab - x, axis=1), np.linalg.norm(cba - y, axis=1))
    magnitude = np.maximum(np.linalg.norm(ab, axis=1), np.linalg.norm(ba, axis=1))
    # Symmetric half-corrections preserve the two-view mean coordinate offset.
    qa, qb = x - 0.5 * delta, y + 0.5 * delta
    safe = _inside(a, qa, margin) & _inside(b, qb, margin)
    before, oldstd = _patch_similarity(a, b, x, y, policy.patch_size)
    after, newstd = _patch_similarity(a, b, qa, qb, policy.patch_size)
    std = np.minimum(oldstd, newstd)
    eig = np.minimum(_sample(ea, x), _sample(eb, y))
    status = goodab & goodba & scab & scba & safe
    finite = np.isfinite(delta).all(axis=1) & np.isfinite(antisym) & np.isfinite(cycle)
    reject = np.zeros(len(which), np.uint16)
    reject[~status | ~finite] |= 2
    reject[(eig < policy.min_eigenvalue) | (std < policy.minimum_patch_std)] |= 4
    reject[(antisym > policy.maximum_antisymmetry_error) | (cycle > policy.maximum_cycle_error)] |= 8
    reject[(magnitude > policy.maximum_pair_displacement) | (np.linalg.norm(delta, axis=1) > policy.maximum_pair_displacement)] |= 16
    reject[(after < policy.minimum_zncc) | (after - before < policy.minimum_zncc_gain)] |= 32
    quality = np.clip((after - policy.minimum_zncc) / (1.0 - policy.minimum_zncc), 0, 1)
    quality *= np.exp(-0.5 * (antisym / policy.maximum_antisymmetry_error) ** 2
                      -0.5 * (cycle / policy.maximum_cycle_error) ** 2)
    weight = np.clip(quality, policy.minimum_edge_weight, 1).astype(np.float32)
    values = dict(delta=delta, zncc_before=before, zncc_after=after, patch_std=std,
                  texture_eigenvalue=eig, antisymmetry_error=antisym, cycle_error=cycle,
                  pair_displacement=magnitude, weight=weight, accepted=reject == 0,
                  rejection_mask=reject)
    for key, value in values.items():
        result[key][which] = value
    return result


def _candidate_edges(obs: np.ndarray, camera_ids: np.ndarray, policy: Policy) -> np.ndarray:
    # Sequence distance is based on original sorted-image IDs, not registration order.
    global_ids = camera_ids[obs["camera"]]
    order = np.lexsort((global_ids, obs["track"]))
    tracks = obs["track"][order]
    starts = np.r_[0, np.flatnonzero(tracks[1:] != tracks[:-1]) + 1, len(order)]
    edges = []
    for lo, hi in zip(starts[:-1], starts[1:]):
        rows = order[lo:hi]
        for i in range(len(rows)):
            for j in range(i + 1, min(i + 1 + policy.neighbors_each_direction, len(rows))):
                if int(global_ids[rows[j]]) - int(global_ids[rows[i]]) <= policy.camera_sequence_gap:
                    edges.append((int(rows[i]), int(rows[j])))
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


def _solve_graph(edges: np.ndarray, differences: np.ndarray, weights: np.ndarray,
                 count: int, policy: Policy) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve independent connected components, each with zero-mean displacement."""
    displacement = np.zeros((count, 2), np.float64)
    support = np.zeros(count, np.uint16)
    quality_sum = np.zeros(count, np.float64)
    parent = np.arange(count)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for a, b in edges:
        aa, bb = root(int(a)), root(int(b))
        if aa != bb:
            parent[bb] = aa
    components: dict[int, list[int]] = defaultdict(list)
    for k, (a, _) in enumerate(edges):
        components[root(int(a))].append(k)
    too_large = singular = 0
    fit_residual = np.zeros(len(edges), np.float32)
    component_means = []
    for edge_rows in components.values():
        er = np.asarray(edge_rows)
        nodes, ij = np.unique(edges[er], return_inverse=True)
        ij = ij.reshape(-1, 2)
        wa = weights[er].astype(np.float64)
        target = differences[er].astype(np.float64)
        robust = np.ones(len(er))
        solution = np.zeros((len(nodes), 2))
        try:
            for _ in range(policy.graph_irls_iterations):
                w = wa * robust
                matrix = np.eye(len(nodes)) * policy.graph_regularization
                rhs = np.zeros((len(nodes), 2))
                aa, bb = ij.T
                np.add.at(matrix, (aa, aa), w)
                np.add.at(matrix, (bb, bb), w)
                np.add.at(matrix, (aa, bb), -w)
                np.add.at(matrix, (bb, aa), -w)
                np.add.at(rhs, aa, -w[:, None] * target)
                np.add.at(rhs, bb, w[:, None] * target)
                solution = np.linalg.solve(matrix, rhs)
                solution -= solution.mean(axis=0, keepdims=True)
                residual = np.linalg.norm(solution[bb] - solution[aa] - target, axis=1)
                robust = np.minimum(1.0, policy.graph_huber / np.maximum(residual, 1e-12))
        except np.linalg.LinAlgError:
            singular += 1
            continue
        fit_residual[er] = residual.astype(np.float32)
        # Reject the entire component instead of clipping individual coordinates;
        # clipping would break the explicitly retained translation gauge.
        if not np.isfinite(solution).all() or np.linalg.norm(solution, axis=1).max() > policy.maximum_observation_displacement:
            too_large += 1
            continue
        displacement[nodes] = solution
        for a, b, w in zip(edges[er, 0], edges[er, 1], wa * robust):
            support[a] += 1
            support[b] += 1
            quality_sum[a] += w
            quality_sum[b] += w
        component_means.append(float(np.max(np.abs(solution.mean(axis=0)))))
    quality = quality_sum / np.maximum(support, 1)
    info = dict(components=len(components), rejected_displacement_components=too_large,
                rejected_singular_components=singular,
                max_component_mean_displacement=max(component_means, default=0.0),
                edge_graph_fit_residual_px=fit_residual, quality=quality)
    return displacement, support, info


def refine_observations(sfm: Path, images: Path, output: Path) -> dict:
    """Write refined observations and diagnostics to a NEW directory.

    Native output layout is unchanged. Only u/v may differ. Sidecars are aligned
    with the original row order: observation_sigma.bin / observation_confidence.bin
    contain little-endian float32. This function never launches BA or prunes data.
    """
    sfm, images, output = Path(sfm).resolve(), Path(images).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite refinement output: {output}")
    if not (sfm / "observations.bin").exists():
        raise FileNotFoundError(sfm / "observations.bin")
    started = time.monotonic()
    policy = POLICY
    obs_path = sfm / "observations.bin"
    if obs_path.stat().st_size % OBS.itemsize:
        raise ValueError("Invalid native observations.bin length")
    obs = np.fromfile(obs_path, OBS)
    ids = np.fromfile(sfm / "camera_ids.bin", "<u4")
    if not len(obs) or not len(ids) or int(obs["camera"].max()) >= len(ids):
        raise ValueError("Empty observations or invalid local camera index")
    if not np.isfinite(np.c_[obs["u"], obs["v"]]).all():
        raise ValueError("Nonfinite observation coordinates")
    if len(np.unique(np.c_[obs["track"], obs["camera"]], axis=0)) != len(obs):
        raise ValueError("A track contains duplicate camera observations")
    paths = _image_paths(sfm, images, ids)
    input_hashes = {name: _sha(sfm / name) for name in ("observations.bin", "camera_ids.bin")}
    output.mkdir(parents=True)
    cv2.setNumThreads(policy.cv_threads)
    cache = _Images(paths, policy)
    points = np.c_[obs["u"], obs["v"]].astype(np.float32) - 0.5
    edges = _candidate_edges(obs, ids, policy)
    diagnostic: dict[str, np.ndarray] = {}
    if len(edges):
        cameras = np.c_[obs["camera"][edges[:, 0]], obs["camera"][edges[:, 1]]]
        order = np.lexsort((cameras[:, 1], cameras[:, 0]))
        pairs = cameras[order]
        starts = np.r_[0, np.flatnonzero(np.any(pairs[1:] != pairs[:-1], axis=1)) + 1, len(order)]
        for pair_index, (lo, hi) in enumerate(zip(starts[:-1], starts[1:])):
            er = order[lo:hi]
            ca, cb = map(int, pairs[lo])
            a, ea = cache.get(ca)
            b, eb = cache.get(cb)
            if a.shape != b.shape:
                raise ValueError("LK refinement requires identically sized images")
            evidence = _pair_evidence(a, b, ea, eb, points[edges[er, 0]], points[edges[er, 1]], policy)
            for key, value in evidence.items():
                if key not in diagnostic:
                    diagnostic[key] = np.zeros((len(edges),) + value.shape[1:], dtype=value.dtype)
                diagnostic[key][er] = value
            if pair_index % 20 == 0 or pair_index + 1 == len(starts) - 1:
                print(f"image refinement: pair {pair_index + 1}/{len(starts) - 1}", flush=True)
    else:
        diagnostic = dict(accepted=np.zeros(0, bool), delta=np.zeros((0, 2), np.float32),
                          weight=np.zeros(0, np.float32), rejection_mask=np.zeros(0, np.uint16))
    accepted = diagnostic["accepted"]
    delta, support, solve_info = _solve_graph(edges[accepted], diagnostic["delta"][accepted],
                                              diagnostic["weight"][accepted], len(obs), policy)
    refined = obs.copy()
    refined["u"] = (obs["u"].astype(np.float64) + delta[:, 0]).astype(np.float32)
    refined["v"] = (obs["v"].astype(np.float64) + delta[:, 1]).astype(np.float32)
    actual_delta = np.c_[refined["u"] - obs["u"], refined["v"] - obs["v"]]
    modified = np.any(actual_delta != 0, axis=1)
    norm = np.linalg.norm(actual_delta, axis=1)
    if norm.max(initial=0) > policy.maximum_observation_displacement + 0.001:
        raise AssertionError("Float32 output violates displacement bound")
    for key in ("camera", "track", "aux", "depth", "size"):
        if refined[key].tobytes() != obs[key].tobytes():
            raise AssertionError(f"Identity or metadata changed: {key}")
    quality = np.clip(solve_info.pop("quality"), 0, 1)
    confidence = np.full(len(obs), 0.25, dtype="<f4")
    sigma = np.ones(len(obs), dtype="<f4")
    confidence[modified] = 0.25 + 0.75 * quality[modified]
    sigma[modified] = np.clip(1.0 - 0.4 * quality[modified], policy.sigma_floor, 1.0)
    graph_residual = np.full(len(edges), np.nan, np.float32)
    graph_residual[accepted] = solve_info.pop("edge_graph_fit_residual_px")
    refined.tofile(output / "observations_refined.bin")
    confidence.tofile(output / "observation_confidence.bin")
    sigma.tofile(output / "observation_sigma.bin")
    np.savez_compressed(output / "diagnostics.npz", row_index=np.arange(len(obs)),
                        camera=obs["camera"], track=obs["track"],
                        original_uv=np.c_[obs["u"], obs["v"]], refined_uv=np.c_[refined["u"], refined["v"]],
                        displacement_px=actual_delta, support=support, modified=modified,
                        confidence=confidence, sigma=sigma, edge_rows=edges,
                        edge_graph_fit_residual_px=graph_residual,
                        **{f"edge_{key}": value for key, value in diagnostic.items()})
    if input_hashes != {name: _sha(sfm / name) for name in input_hashes}:
        raise AssertionError("Input native model changed during refinement")
    rejection = diagnostic["rejection_mask"]
    summary = dict(
        complete=True, method="symmetric_image_track_refinement", version=1,
        source_sfm=str(sfm), images=str(images), policy=asdict(policy),
        coordinates="Native COLMAP +0.5 centers; subtract 0.5 for OpenCV only",
        uses_projection_or_3d=False, changes_track_membership=False,
        input_observations=len(obs), output_observations=len(refined),
        candidate_edges=len(edges), image_verified_edges=int(accepted.sum()),
        modified_observations=int(modified.sum()), unchanged_observations=int((~modified).sum()),
        observations_with_image_support=int((support > 0).sum()),
        displacement_mean_modified_px=float(norm[modified].mean()) if modified.any() else 0,
        displacement_median_modified_px=float(np.median(norm[modified])) if modified.any() else 0,
        displacement_p95_modified_px=float(np.quantile(norm[modified], .95)) if modified.any() else 0,
        displacement_max_px=float(norm.max(initial=0)),
        edge_rejection_gate_counts={name: int(np.count_nonzero(rejection & bit)) for name, bit in
                                   (("border", 1), ("lk_or_nonfinite", 2), ("texture", 4),
                                    ("antisymmetry_or_cycle", 8), ("displacement", 16), ("patch", 32))},
        graph=solve_info, input_sha256=input_hashes,
        image_sha256=cache.hashes, module_sha256=_sha(Path(__file__)),
        output_sha256={name: _sha(output / name) for name in
                       ("observations_refined.bin", "observation_confidence.bin", "observation_sigma.bin")},
        confidence_note="Bounded heuristic from image correspondence quality only; not calibrated uncertainty. Unmodified sigma=1.",
        limitations=["Local translation approximation; surface deformation, occlusion, and specularity can defeat the gates.",
                     "Only neighbors within three sorted camera IDs are used; this assumes sequential acquisition.",
                     "Trackwise zero-mean gauge cannot remove a common localization bias.",
                     "Classical LK/graph integration, not a verified reproduction of PixSfM or Metashape and not a novelty claim.",
                     "No improvement of geometric accuracy follows without held-out evaluation."],
        elapsed_seconds=time.monotonic() - started)
    _save_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfm", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = refine_observations(args.sfm, args.images, args.output)
    print(json.dumps({k: result[k] for k in ("modified_observations", "image_verified_edges", "elapsed_seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
