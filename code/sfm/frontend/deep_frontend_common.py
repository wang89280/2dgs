from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

_ORT_CAPI_DIR = Path(sys.prefix) / "Lib" / "site-packages" / "onnxruntime" / "capi"
if _ORT_CAPI_DIR.exists():
    os.environ["PATH"] = str(_ORT_CAPI_DIR) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(_ORT_CAPI_DIR))

import onnxruntime as ort
import pycolmap


@dataclass
class FeatureSet:
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray | None
    image_size: tuple[int, int]
    scales: np.ndarray | None = None
    orientations: np.ndarray | None = None

def make_session(model_path: Path, provider: str) -> ort.InferenceSession:
    provider_map = {
        "cpu": ["CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        "auto": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }
    return ort.InferenceSession(str(model_path.resolve()), providers=provider_map[provider])

def resize_for_network(image: np.ndarray, max_size: int, divisor: int = 32) -> tuple[np.ndarray, float, float]:
    h, w = image.shape[:2]
    scale = 1.0
    if max_size > 0 and max(h, w) > max_size:
        scale = max_size / float(max(h, w))
    new_w = max(divisor, int(round(w * scale)) // divisor * divisor)
    new_h = max(divisor, int(round(h * scale)) // divisor * divisor)
    if new_w == w and new_h == h:
        return image, 1.0, 1.0
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, w / float(new_w), h / float(new_h)

def normalize_descriptors(desc: np.ndarray) -> np.ndarray:
    desc = np.asarray(desc, dtype=np.float32)
    norm = np.linalg.norm(desc, axis=1, keepdims=True)
    return desc / np.maximum(norm, 1e-12)

def sort_and_limit(features: FeatureSet, max_keypoints: int) -> FeatureSet:
    n = len(features.keypoints)
    if n == 0:
        return features
    if features.scores is not None:
        order = np.argsort(-features.scores)
    else:
        order = np.arange(n)
    if max_keypoints > 0:
        order = order[:max_keypoints]
    scales = features.scales[order] if features.scales is not None else None
    orientations = features.orientations[order] if features.orientations is not None else None
    scores = features.scores[order] if features.scores is not None else None
    return FeatureSet(
        keypoints=np.ascontiguousarray(features.keypoints[order], dtype=np.float32),
        descriptors=np.ascontiguousarray(features.descriptors[order], dtype=np.float32),
        scores=np.ascontiguousarray(scores, dtype=np.float32) if scores is not None else None,
        image_size=features.image_size,
        scales=np.ascontiguousarray(scales, dtype=np.float32) if scales is not None else None,
        orientations=np.ascontiguousarray(orientations, dtype=np.float32) if orientations is not None else None,
    )

def extract_aliked(session: ort.InferenceSession, image_path: Path, args: argparse.Namespace) -> FeatureSet:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    orig_h, orig_w = bgr.shape[:2]
    resized, sx, sy = resize_for_network(bgr, args.max_image_size, 32)
    h, w = resized.shape[:2]
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = np.transpose(rgb[None], (0, 3, 1, 2))
    keypoints, descriptors, scores = session.run(
        None,
        {
            "image": image,
            "max_keypoints": np.array(args.max_keypoints, dtype=np.int64),
            "min_score": np.array(args.min_score, dtype=np.float32),
        },
    )
    keypoints = keypoints.reshape(-1, 2).astype(np.float32)
    descriptors = descriptors.reshape(-1, descriptors.shape[-1]).astype(np.float32)
    scores = scores.reshape(-1).astype(np.float32)
    if keypoints.size and keypoints.max() <= 2.0 and keypoints.min() >= -2.0:
        keypoints[:, 0] = (keypoints[:, 0] + 1.0) * 0.5 * (w - 1) + 0.5
        keypoints[:, 1] = (keypoints[:, 1] + 1.0) * 0.5 * (h - 1) + 0.5
    keypoints[:, 0] *= sx
    keypoints[:, 1] *= sy
    return sort_and_limit(
        FeatureSet(keypoints, normalize_descriptors(descriptors), scores, (orig_w, orig_h)),
        args.max_keypoints,
    )

def match_lightglue_2out(session: ort.InferenceSession, f0: FeatureSet, f1: FeatureSet, _: argparse.Namespace) -> np.ndarray:
    out = session.run(
        None,
        {
            "kpts0": f0.keypoints[None].astype(np.float32),
            "kpts1": f1.keypoints[None].astype(np.float32),
            "desc0": f0.descriptors[None].astype(np.float32),
            "desc1": f1.descriptors[None].astype(np.float32),
            "image_size0": np.array([[f0.image_size[0], f0.image_size[1]]], dtype=np.float32),
            "image_size1": np.array([[f1.image_size[0], f1.image_size[1]]], dtype=np.float32),
        },
    )
    matches0 = out[0]
    if matches0.ndim == 2 and matches0.shape[1] == 2:
        return matches0.astype(np.uint32)
    idx0 = np.arange(matches0.shape[0], dtype=np.int64)
    valid = matches0 >= 0
    return np.stack([idx0[valid], matches0[valid]], axis=1).astype(np.uint32)

def generate_pairs(names: list[str], pairing: str, sequential_window: int, loop_closure: bool) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    n = len(names)
    if pairing == "exhaustive":
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((names[i], names[j]))
        return pairs
    for i in range(n):
        max_j = min(n, i + sequential_window + 1)
        for j in range(i + 1, max_j):
            pairs.append((names[i], names[j]))
    if loop_closure and n > sequential_window + 1:
        for i in range(n - sequential_window, n):
            for j in range(0, min(sequential_window, n)):
                if i != j:
                    a, b = sorted((i, j))
                    pair = (names[a], names[b])
                    if pair not in pairs:
                        pairs.append(pair)
    return pairs

def write_keypoints(database_path: Path, image_ids: dict[str, int], features: dict[str, FeatureSet]) -> None:
    db = pycolmap.Database.open(str(database_path))
    try:
        with pycolmap.DatabaseTransaction(db):
            for name, feature in features.items():
                db.write_keypoints(image_ids[name], np.ascontiguousarray(feature.keypoints, dtype=np.float32))
    finally:
        db.close()

def write_matches(
    database_path: Path,
    image_ids: dict[str, int],
    matched_pairs: list[tuple[str, str, np.ndarray]],
) -> None:
    db = pycolmap.Database.open(str(database_path))
    try:
        with pycolmap.DatabaseTransaction(db):
            for name0, name1, matches in matched_pairs:
                if len(matches) == 0:
                    continue
                db.write_matches(
                    image_ids[name0],
                    image_ids[name1],
                    np.ascontiguousarray(matches, dtype=np.uint32),
                )
    finally:
        db.close()

def write_pairs_file(path: Path, matched_pairs: list[tuple[str, str, np.ndarray]]) -> None:
    lines = [f"{a} {b}\n" for a, b, matches in matched_pairs if len(matches) > 0]
    path.write_text("".join(lines), encoding="utf-8")
