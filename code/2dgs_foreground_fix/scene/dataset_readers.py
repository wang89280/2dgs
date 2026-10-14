#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import shutil
import subprocess
import tempfile
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: Image.Image = None
    mask_path: str = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, masks_folder=None, use_masks=False):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        mask = None
        mask_path = None
        if use_masks:
            if masks_folder is None:
                raise ValueError("use_masks=True but no masks folder was provided.")
            mask_path = os.path.join(masks_folder, f"{image_name}.png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Could not find training mask for image '{image_name}' at '{mask_path}'."
                )
            mask = Image.open(mask_path).convert("L")

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              mask=mask, mask_path=mask_path)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def getColmapSparseDir(path):
    sparse_zero_dir = os.path.join(path, "sparse", "0")
    sparse_root_dir = os.path.join(path, "sparse")

    if os.path.exists(sparse_zero_dir):
        return sparse_zero_dir
    if os.path.exists(sparse_root_dir):
        return sparse_root_dir
    raise FileNotFoundError(
        f"Could not find COLMAP sparse directory under {path}. "
        "Expected either sparse/0 or sparse."
    )

def _getColmapCameraModels(sparse_dir):
    """Return camera model names without assuming that they are undistorted.

    The legacy text-model reader asserts PINHOLE, so use a small parser here
    when deciding whether the model first needs to be undistorted.
    """
    binary_path = os.path.join(sparse_dir, "cameras.bin")
    if os.path.isfile(binary_path):
        return [camera.model for camera in read_intrinsics_binary(binary_path).values()]

    text_path = os.path.join(sparse_dir, "cameras.txt")
    models = []
    with open(text_path, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                models.append(line.split()[1])
    if not models:
        raise RuntimeError(f"No camera definitions found in {text_path}")
    return models

def _areUndistortedColmapModels(camera_models):
    return all(model in ("PINHOLE", "SIMPLE_PINHOLE") for model in camera_models)

def _hasColmapModel(sparse_dir):
    return (os.path.isfile(os.path.join(sparse_dir, "cameras.bin")) and
            os.path.isfile(os.path.join(sparse_dir, "images.bin")) and
            os.path.isfile(os.path.join(sparse_dir, "points3D.bin"))) or \
           (os.path.isfile(os.path.join(sparse_dir, "cameras.txt")) and
            os.path.isfile(os.path.join(sparse_dir, "images.txt")) and
            os.path.isfile(os.path.join(sparse_dir, "points3D.txt")))

def prepareColmapFor2DGS(path, images):
    """Return image and sparse-model paths that 2DGS can load.

    COLMAP camera models with distortion (for example SIMPLE_RADIAL) are not
    supported by the rasterizer loader.  When one is found, run COLMAP's image
    undistorter once and cache the result next to the original image directory:

        <scene>/images_undistorted/
        <scene>/sparse_undistorted/0/

    The original images and sparse reconstruction are never changed.
    """
    reading_dir = "images" if images is None else images
    images_folder = os.path.join(path, reading_dir)
    sparse_dir = getColmapSparseDir(path)
    camera_models = _getColmapCameraModels(sparse_dir)
    if _areUndistortedColmapModels(camera_models):
        return images_folder, sparse_dir

    image_dir_name = os.path.basename(os.path.normpath(reading_dir))
    undistorted_images = os.path.join(path, f"{image_dir_name}_undistorted")
    undistorted_sparse = os.path.join(path, "sparse_undistorted", "0")

    if _hasColmapModel(undistorted_sparse) and os.path.isdir(undistorted_images):
        cached_models = _getColmapCameraModels(undistorted_sparse)
        if _areUndistortedColmapModels(cached_models):
            print(f"Using cached undistorted COLMAP data: {undistorted_images}")
            return undistorted_images, undistorted_sparse

    cache_root = os.path.dirname(undistorted_sparse)
    if os.path.exists(undistorted_images) or os.path.exists(cache_root):
        raise RuntimeError(
            "Found an incomplete or incompatible undistortion cache at "
            f"'{undistorted_images}' or '{cache_root}'. Rename or remove those "
            "generated directories before retrying; the original dataset is untouched."
        )

    colmap = shutil.which("colmap")
    if colmap is None:
        models = sorted(set(camera_models))
        raise RuntimeError(
            "The COLMAP reconstruction uses distorted camera model(s) "
            f"{models}, but `colmap` was not found in PATH. Install COLMAP or "
            "provide an undistorted PINHOLE/SIMPLE_PINHOLE reconstruction."
        )
    if not os.path.isdir(images_folder):
        raise FileNotFoundError(f"COLMAP image folder does not exist: {images_folder}")

    models = sorted(set(camera_models))
    print("Detected distorted COLMAP camera model(s) {}. "
          "Generating cached undistorted images and sparse model...".format(models))
    temp_dir = tempfile.mkdtemp(prefix=".2dgs_undistort_", dir=path)
    try:
        subprocess.run([
            colmap, "image_undistorter",
            "--image_path", images_folder,
            "--input_path", sparse_dir,
            "--output_path", temp_dir,
            "--output_type", "COLMAP",
        ], check=True)

        temp_images = os.path.join(temp_dir, "images")
        temp_sparse = os.path.join(temp_dir, "sparse")
        if not os.path.isdir(temp_images) or not os.path.isdir(temp_sparse):
            raise RuntimeError("COLMAP image_undistorter did not produce images and sparse outputs.")

        shutil.move(temp_images, undistorted_images)
        shutil.move(temp_sparse, cache_root)
        os.makedirs(undistorted_sparse, exist_ok=True)
        for file_name in os.listdir(cache_root):
            source = os.path.join(cache_root, file_name)
            if os.path.isfile(source):
                shutil.move(source, os.path.join(undistorted_sparse, file_name))

        if not (_hasColmapModel(undistorted_sparse) and os.path.isdir(undistorted_images)):
            raise RuntimeError("Could not create a complete cached undistorted COLMAP dataset.")
        cached_models = _getColmapCameraModels(undistorted_sparse)
        if not _areUndistortedColmapModels(cached_models):
            raise RuntimeError("COLMAP undistortion did not produce a PINHOLE-compatible camera model.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"Saved undistorted images to: {undistorted_images}")
    return undistorted_images, undistorted_sparse

def readColmapSceneInfo(path, images, eval, llffhold=8, use_masks=False, mask_dir="masks_train"):
    images_folder, sparse_dir = prepareColmapFor2DGS(path, images)
    try:
        cameras_extrinsic_file = os.path.join(sparse_dir, "images.bin")
        cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(sparse_dir, "images.txt")
        cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    masks_folder = os.path.join(path, mask_dir) if use_masks else None
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=images_folder,
        masks_folder=masks_folder,
        use_masks=False,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # Split first: evaluation masks are neither generated nor loaded.
    if use_masks:
        loaded_train = []
        for camera in train_cam_infos:
            mask_path = os.path.join(masks_folder, camera.image_name + '.png')
            if not os.path.isfile(mask_path):
                raise FileNotFoundError('Missing training mask: ' + mask_path)
            mask = Image.open(mask_path).convert('L')
            if mask.size != camera.image.size:
                raise ValueError('Training mask/image dimensions differ: ' + mask_path)
            loaded_train.append(camera._replace(mask=mask, mask_path=mask_path))
        train_cam_infos = loaded_train
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(sparse_dir, "points3D.ply")
    bin_path = os.path.join(sparse_dir, "points3D.bin")
    txt_path = os.path.join(sparse_dir, "points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            mask=None, mask_path=None))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
