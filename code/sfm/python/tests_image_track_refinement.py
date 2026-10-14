"""Synthetic image-evidence and identity tests; no benchmark data or test metrics."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from image_track_refinement import POLICY, _solve_graph, refine_observations
from model_io import OBS


class ImageTrackRefinementTests(unittest.TestCase):
    def test_graph_sign_gauge_and_component_rejection(self):
        edges = np.array([[0, 1], [1, 2], [0, 2], [3, 4]])
        truth = np.array([[-.4, .2], [.1, -.3], [.3, .1], [-3., 0], [3., 0]])
        differences = truth[edges[:, 1]] - truth[edges[:, 0]]
        policy = replace(POLICY, graph_regularization=1e-8)
        delta, support, info = _solve_graph(edges, differences, np.ones(4), 6, policy)
        np.testing.assert_allclose(delta[:3], truth[:3], atol=1e-7)
        np.testing.assert_allclose(delta[:3].mean(axis=0), 0, atol=1e-12)
        np.testing.assert_array_equal(delta[3:], 0)
        np.testing.assert_array_equal(support[3:], 0)
        self.assertEqual(info['rejected_displacement_components'], 1)

    def test_known_subpixel_errors_and_wrong_matches(self):
        rng = np.random.default_rng(17842)
        with tempfile.TemporaryDirectory(prefix='image_refinement_synthetic_') as folder:
            root = Path(folder)
            sfm, images, output = root/'sfm', root/'images', root/'output'
            sfm.mkdir()
            images.mkdir()
            # Smoothed random texture supplies independent, isotropic image evidence.
            texture = rng.uniform(0, 255, (360, 480)).astype(np.float32)
            texture = cv2.GaussianBlur(texture, (0, 0), .85)
            texture = np.clip((texture - 128) * 2.5 + 128, 0, 255).astype(np.uint8)
            shifts = np.array([[0, 0], [.8, -.45], [1.7, -.9], [2.6, -1.3], [3.5, -1.8]])
            for camera, shift in enumerate(shifts):
                transform = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
                frame = cv2.warpAffine(texture, transform, (480, 360), flags=cv2.INTER_LINEAR)
                # A genuinely textureless, identical patch should remain untouched.
                frame[5:35, 5:35] = 120
                cv2.imwrite(str(images/f'{camera+1:04d}.png'), frame)
            grid = np.array([(x, y) for y in (70, 120, 170, 220, 270) for x in (65, 115, 165, 215, 265, 315, 365, 415)], dtype=float)
            truth = (grid[:, None, :] + shifts[None, :, :]).reshape(-1, 2) + .5
            jitter = rng.normal(0, .55, (len(grid), 5, 2))
            jitter -= jitter.mean(axis=1, keepdims=True)
            noisy = truth + jitter.reshape(-1, 2)
            # Last track has entirely wrong inter-image associations, not just noise.
            wrong = np.array([[65, 100], [350, 60], [140, 250], [410, 190], [260, 310]], float) + .5
            xy = np.r_[noisy, wrong, np.tile([[20.5, 20.5]], (5, 1))]
            obs = np.zeros(len(xy), dtype=OBS)
            obs['camera'] = np.tile(np.arange(5), len(grid)+2)
            obs['track'] = np.repeat(np.arange(len(grid)+2), 5)
            obs['aux'] = np.arange(len(obs)) * 777
            obs['depth'] = np.arange(len(obs), dtype=np.float32) * .01
            obs['size'] = 1
            obs['u'], obs['v'] = xy.T
            obs.tofile(sfm/'observations.bin')
            np.arange(5, dtype='<u4').tofile(sfm/'camera_ids.bin')
            (sfm/'cameras.xml').write_text('<document><cameras>' + ''.join(
                f'<camera id="{i}" label="{i+1:04d}"/>' for i in range(5)) + '</cameras></document>')
            original_bytes = (sfm/'observations.bin').read_bytes()
            result = refine_observations(sfm, images, output)
            refined = np.fromfile(output/'observations_refined.bin', OBS)
            refined_xy = np.c_[refined['u'], refined['v']]
            before = float(np.linalg.norm(noisy-truth, axis=1).mean())
            after = float(np.linalg.norm(refined_xy[:len(truth)]-truth, axis=1).mean())
            print(json.dumps(dict(synthetic_before_mean_px=before, synthetic_after_mean_px=after,
                                  modified=result['modified_observations']), indent=2))
            self.assertLess(after, .55 * before)
            self.assertGreater(result['modified_observations'], .8 * len(truth))
            np.testing.assert_array_equal(refined_xy[-10:], np.c_[obs['u'], obs['v']][-10:])
            for key in ('camera', 'track', 'aux', 'depth', 'size'):
                self.assertEqual(refined[key].tobytes(), obs[key].tobytes())
            self.assertEqual((sfm/'observations.bin').read_bytes(), original_bytes)
            sigma = np.fromfile(output/'observation_sigma.bin', '<f4')
            self.assertTrue(np.all((sigma >= POLICY.sigma_floor) & (sigma <= 1)))
            np.testing.assert_array_equal(sigma[-10:], 1)
            with np.load(output/'diagnostics.npz') as diagnostic:
                self.assertLessEqual(float(np.linalg.norm(diagnostic['displacement_px'], axis=1).max()), 2.001)
            with self.assertRaises(FileExistsError):
                refine_observations(sfm, images, output)


if __name__ == '__main__':
    unittest.main()
