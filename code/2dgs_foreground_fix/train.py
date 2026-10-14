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
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.foreground_stability import mask_regions, stability_losses
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def resize_mask_like(mask, image):
    if mask is None:
        return None
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[-2:] != image.shape[-2:]:
        mask = F.interpolate(mask.unsqueeze(0), size=image.shape[-2:], mode="nearest").squeeze(0)
    return mask.clamp(0.0, 1.0)


def masked_l1_loss(image, gt_image, mask):
    diff = torch.abs(image - gt_image)
    denom = mask.sum() * image.shape[0] + 1e-8
    return (diff * mask).sum() / denom

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, autosave_iterations):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    resumed_state = None
    if checkpoint:
        checkpoint_data = torch.load(checkpoint)
        model_params, first_iter = checkpoint_data[:2]
        if len(checkpoint_data) > 2:
            resumed_state = checkpoint_data[2]
        gaussians.restore(model_params, opt)

    if opt.foreground_stability:
        if not dataset.use_masks:
            raise ValueError('Foreground stability requires training masks')
        for cam in scene.getTrainCameras():
            mask = resize_mask_like(cam.gt_mask, cam.original_image)
            if mask is None:
                raise ValueError('Missing training mask: ' + cam.image_name)
            # Keep mask tensors on CPU alongside RGB and transfer one view at a time.
            cam.stability_regions = tuple(x.cpu() for x in mask_regions(mask.cpu()))
        assert all(c.gt_mask is None for c in scene.getTestCameras()), 'Evaluation masks must not be loaded'
    with open(os.path.join(scene.model_path, 'optimization_args.json'), 'w') as f:
        json.dump(vars(opt), f, indent=2)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    if resumed_state:
        random.setstate(resumed_state['python_rng'])
        np.random.set_state(resumed_state['numpy_rng'])
        torch.set_rng_state(resumed_state['torch_rng'])
        torch.cuda.set_rng_state_all(resumed_state['cuda_rng'])
        by_name = {c.image_name: c for c in scene.getTrainCameras()}
        viewpoint_stack = [by_name[n] for n in resumed_state['view_stack']]
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        if getattr(dataset, "use_masks", False) and not opt.foreground_stability:
            mask = resize_mask_like(viewpoint_cam.gt_mask, gt_image)
            if mask is None:
                raise RuntimeError(
                    f"--use_masks is enabled but camera '{viewpoint_cam.image_name}' does not have a loaded mask."
                )
            mask = mask.to(gt_image.device)
            Ll1 = masked_l1_loss(image, gt_image, mask)
            masked_render = image * mask
            masked_gt = gt_image * mask
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(masked_render, masked_gt))
        else:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        silhouette_loss = image.new_zeros(())
        if opt.foreground_stability:
            fg, bg = (region.to(image.device) for region in viewpoint_cam.stability_regions)
            silhouette_loss, normal_loss, dist_loss, alpha_bg, alpha_fg_error = stability_losses(
                render_pkg, fg, bg, iteration, opt)

        # loss
        total_loss = loss + dist_loss + normal_loss + silhouette_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError('Non-finite loss at iteration ' + str(iteration))
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            autosave_due = autosave_iterations > 0 and iteration % autosave_iterations == 0
            save_gaussians = iteration in saving_iterations or autosave_due or iteration == opt.iterations
            save_checkpoint = iteration in checkpoint_iterations or autosave_due or iteration == opt.iterations

            if save_gaussians:
                save_reason = "automatic" if autosave_due else "requested/final"
                print("\n[ITER {}] Saving Gaussians ({})".format(iteration, save_reason))
                scene.save(iteration)


            # Densification
            if opt.foreground_stability:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if opt.foreground_stability and size_threshold:
                        size_threshold = opt.max_screen_radius_fraction * viewpoint_cam.image_width
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if opt.foreground_stability and iteration >= opt.densify_until_iter and iteration % opt.stability_prune_interval == 0 and iteration < opt.iterations:
                removed = gaussians.prune_unstable(scene.cameras_extent,
                    opt.max_screen_radius_fraction * viewpoint_cam.image_width)
                print('\nSTABILITY_PRUNE', iteration, removed, flush=True)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if save_checkpoint:
                save_reason = "automatic" if autosave_due else "requested/final"
                print("\n[ITER {}] Saving Checkpoint ({})".format(iteration, save_reason))
                rng_state = dict(python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    view_stack=[c.image_name for c in viewpoint_stack])
                torch.save((gaussians.capture(), iteration, rng_state), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            if opt.foreground_stability and iteration % 100 == 0:
                progress = dict(iteration=iteration,gaussians=len(gaussians.get_xyz),
                    rgb_loss=float(loss),normal_loss=float(normal_loss),silhouette_loss=float(silhouette_loss),
                    training_view=viewpoint_cam.image_name,background_alpha=float(alpha_bg),
                    foreground_alpha_error=float(alpha_fg_error),max_scale=float(gaussians.get_scaling.max()))
                path=os.path.join(scene.model_path, 'stability_progress.json')
                with open(path+'.tmp','w') as f:json.dump(progress,f,indent=2)
                os.replace(path+'.tmp',path)
                if iteration % 1000 == 0: print('\nSTABILITY',json.dumps(progress),flush=True)

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument(
        "--autosave_iterations",
        type=int,
        default=5_000,
        help="Automatically save point_cloud/iteration_<N>/point_cloud.ply and chkpnt<N>.pth every N iterations. Set 0 to disable periodic autosaves; the final iteration is always saved.",
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.autosave_iterations)

    # All done
    print("\nTraining complete.")
