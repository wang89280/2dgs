"""Independent input/final audit for the frozen silhouette-SfM experiment.

Run in WSL so image and mask symlinks resolve. --inputs-only is safe during
training; the default additionally requires completed models and visual review.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from audit_sfm_render_pose_delivery import local, read, sha, vertices

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/sfm_surface_init_20261014'
OLD = ROOT / 'outputs/foreground_fix_2dgs_2-1_20261014'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs-only', action='store_true')
    args = parser.parse_args()
    p = read(OUT / 'protocol.json')
    previous = read(OLD / 'protocol.json')
    protected = {**p['protected_sha256'], **p['new_inputs_sha256']}
    for path, expected in protected.items():
        assert sha(path) == expected, path
    code = local('/home/wk/01_code/03_gs/2dgs-foreground-fix-20261014/code')
    for path, expected in p['repair_code_sha256'].items():
        assert sha(code / path) == sha(ROOT / 'code/2dgs_foreground_fix' / path) == expected, path
    for path, expected in p['experiment_code_sha256'].items():
        assert sha(ROOT / path) == expected, path
    for key in ['common_args', 'train_names', 'test_names']:
        assert p[key] == previous[key], key
    assert len(p['train_names']) == 70 and len(p['test_names']) == 10
    assert not set(p['train_names']) & set(p['test_names'])
    datasets = {**p['datasets'], **{m: previous['datasets'][m] for m in ['OFF', 'COLMAP']}}
    for dataset in datasets.values():
        for name, expected in p['image_sha256'].items():
            assert sha(local(dataset) / 'images' / (name + '.png')) == expected
    for method in ['HULL', 'GATE']:
        dataset = local(datasets[method])
        off = local(datasets['OFF'])
        for name in ['cameras.txt', 'images.txt']:
            assert sha(dataset / 'sparse/0' / name) == sha(off / 'sparse/0' / name)
        masks = sorted(q.stem for q in (dataset / 'masks_train').glob('*.png'))
        assert masks == sorted(p['train_names']), masks
        for name in masks:
            assert sha(dataset / 'masks_train' / (name + '.png')) == sha(off / 'masks_train' / (name + '.png'))

    result = np.load(OUT / 'surface_result.npz')
    keep = result['keep']
    original = vertices(local(datasets['OFF']) / 'sparse/0/points3D.ply')
    hull = vertices(local(datasets['HULL']) / 'sparse/0/points3D.ply')
    gate = vertices(local(datasets['GATE']) / 'sparse/0/points3D.ply')
    assert len(hull) == len(gate) == int(keep.sum()) == p['initial_points']
    for name in ['red', 'green', 'blue']:
        assert np.array_equal(hull[name], gate[name]) and np.array_equal(gate[name], original[name][keep])
    for axis, name in enumerate(['x', 'y', 'z']):
        assert np.array_equal(gate[name], original[name][keep])
        assert np.array_equal(hull[name], result['xyz'][keep, axis].astype('f4'))
        assert np.isfinite(hull[name]).all()
    changed = np.any(np.column_stack([hull[n] != gate[n] for n in ['x', 'y', 'z']]), axis=1)
    assert int(changed.sum()) == p['changed_points']
    ids = result['track_ids'][keep]
    assert np.array_equal(ids, np.array(read(OUT / 'HULL_track_ids.json')))
    bridge = read(ROOT / 'outputs/sfm_render_pose_20261014/geometry_preparation.json')
    native = local(bridge['native_source'])
    candidate = OUT / 'sfm_candidate'
    for name in ['camera_ids.bin', 'calibration.bin', 'poses.bin', 'observations.bin', 'active_points.bin']:
        assert sha(candidate / name) == sha(native / name), name
    assert np.array_equal(np.fromfile(candidate / 'initialization_active_points.bin', '<u4'), ids)
    before = np.fromfile(native / 'points.bin', '<f8').reshape(-1, 3)
    after = np.fromfile(candidate / 'points.bin', '<f8').reshape(-1, 3)
    expected_changed = np.zeros(len(before), bool)
    expected_changed[result['track_ids'][keep & ~result['filter_keep']]] = True
    assert np.array_equal(np.any(before != after, axis=1), expected_changed)
    assert np.isfinite(after).all()
    untouched = ['code/2dgs_foreground_fix', 'code/sfm/frontend', 'tools/evaluate_foreground_fix_2dgs.py']
    assert not subprocess.check_output(['git', 'diff', '--name-only', 'ce5c2f5', '--', *untouched], cwd=ROOT).strip()
    audit = dict(passed=True, phase='inputs' if args.inputs_only else 'delivery',
        protected_files=len(protected), dataset_images_hash_checked=len(p['image_sha256']) * len(datasets),
        masks_match_training_split_only=True, code_and_training_args_frozen=True,
        all_80_poses_and_intrinsics_unchanged=True, original_features_tracks_observations_unchanged=True,
        point_ids_count_order_rgb_matched=True, initial_points=len(hull), corrected_points=int(changed.sum()),
        original_models_preserved=True, default_not_replaced=True,
        geometry_claim='Silhouette consistency improved; same-observation native SfM RMSE worsened. No 3D ground truth.')
    if not args.inputs_only:
        assert read(OUT / 'audit.json')['passed']
        report_status = read(OUT / 'report_status.json')
        assert report_status['visual_review_complete']
        assert (OUT / 'visual_review.md').read_text(encoding='utf8').strip()
        model_hashes = {}
        for method in ['HULL', 'GATE']:
            trained = read(OUT / f'trained_{method}.json')
            metrics = read(OUT / 'results' / method / 'metrics.json')
            model = local(trained['model']) / 'point_cloud/iteration_30000/point_cloud.ply'
            assert trained['complete'] and trained['iteration'] == 30000
            assert sha(model) == trained['model_sha256'] == metrics['model_sha256']
            tests = [r['name'] for r in metrics['rows'] if r['split'] == 'test']
            assert tests == p['test_names']
            model_hashes[method] = sha(model)
        for page in [1, 2]:
            assert (OUT / f'comparison_views_{page:02d}.png').is_file()
        audit.update(final_iteration=30000, new_training_runs=2, model_sha256=model_hashes,
            visual_review_complete=True, report_sha256=sha(OUT / '轮廓约束SfM与2DGS报告.md'),
            favorable_on_primary_metrics=report_status['favorable_on_primary_metrics'])
    filename = 'input_delivery_audit.json' if args.inputs_only else 'delivery_audit.json'
    (OUT / filename).write_text(json.dumps(audit, indent=2), encoding='utf8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
