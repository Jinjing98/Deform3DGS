#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import random
import os 
import torch
from random import randint
# from utils.loss_utils import l1_loss
from gaussian_renderer import render_flow as render

import sys
from scene import  Scene
from scene.flexible_deform_model import TissueGaussianModel
# from scene.tool_movement_model import GaussianModelActor
from scene.mis_gaussian_model import MisGaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments import FDMHiddenParams as ModelHiddenParams
from utils.timer import Timer
import torch.nn.functional as F

# import lpips
from utils.scene_utils import render_training_image
from scene.cameras import Camera
from utils.loss_utils import ssim
from train import training_report
from utils.image_utils import save_img_torch, visualize_depth_numpy



to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# def training_misgsmodel(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, extra_mark):
def training_misgsmodel(args,use_streetgs_render = False):
    other_param_dict = None
    dbg_print = True
    dbg_print = False
    remain_redundant = True
    # # have to use streetgs cam model
    from config.argsgroup2cn import perform_args2cfg
    cfg, others = perform_args2cfg(args,
                                    remain_redundant = remain_redundant,
                                    dbg_print = dbg_print,
                                    )
    eval_stree_param,train_stree_param,opt_stree_param,\
            mod_stree_param,data_stree_param,render_stree_param,viewer_stree_param,\
                other_param_dict = others

    os.makedirs(cfg.trained_model_dir,exist_ok=True)
    os.makedirs(cfg.point_cloud_dir,exist_ok=True)
    print('todo maybe support write CN in addtion to args_group..')
    from train import prepare_output_and_logger
    tb_writer = prepare_output_and_logger(model_path=cfg.expname, write_args=args)
    timer = Timer()
    print('todo clean the sceneinfo meta and caminfo meta')
    load_other_obj_meta=True #load within the sceneinfo
    print('////////////////***************///////////')
    print('MisGS reuse the Scene function of deform3dgs (only break it down)')
    scene = Scene(mod_stree_param,
                  load_other_obj_meta=load_other_obj_meta,
                  new_cfg=cfg,
                  )
    
    controller = MisGaussianModel(metadata=scene.getSceneMetaData(),
                                 new_cfg=cfg)#nn.module instance
    scene.gs_init(gaussians_or_controller=controller,
                  reset_camera_extent=mod_stree_param.camera_extent)
    timer.start()
    scene_reconstruction_misgs(cfg = cfg, controller = controller,
                               scene = scene, tb_writer = tb_writer,
                               render_stree_param_for_ori_train_report = render_stree_param,
                                use_streetgs_render = use_streetgs_render,
                               
                               )


def scene_reconstruction_misgs(cfg, controller, scene, tb_writer,
                               render_stree_param_for_ori_train_report = None,
                               use_streetgs_render = False,
                               
                               ):
    
    print('/////////////************/////////////')
    print('Use misGA  scene_recon function:1) entry for more loss 2) render with misGS controller...')
    print('todo make the tissue model inherent from basemodel')
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data
    start_iter = 0
    controller.training_setup()
    try:
        print(f'Loading model from {ckpt_path}')
    except:
        pass
    print(f'Starting from {start_iter}')
    from config.argsgroup2cn import save_cfg
    save_cfg(cfg, cfg.model_path, epoch=start_iter)

    from render_misgs import MisGaussianRenderer
    gaussians_renderer = MisGaussianRenderer(cfg=cfg)
    from gaussian_renderer import render_flow as render

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    #streetgs traing added
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_dict = {}
    progress_bar = tqdm(range(start_iter, training_args.iterations))
    start_iter += 1

    viewpoint_stack = None
    for iteration in range(start_iter, training_args.iterations + 1):
    
        iter_start.record()
        controller.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        if iteration % 500 == 0:
        # if iteration % 1 == 0:
            controller.oneupSHdegree()
            # print('////////////',controller.active_sh_degree)
            # assert 0, controller.tissue.active_sh_degree

        # Every 1000 iterations upsample
        # if iteration % 1000 == 0:
        #     if resolution_scales:  
        #         scale = resolution_scales.pop()

        print('potentially moving out the loop for effeciency.')
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        

        # print("Loading Training Cameras")
        # self.train_camera = scene_info.train_cameras 
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        gt_image = viewpoint_cam.original_image.cuda()

        if hasattr(viewpoint_cam, 'original_mask'):# tissue mask
            mask = viewpoint_cam.original_mask.cuda().bool()
        else:
            mask = torch.ones_like(gt_image[0:1]).bool()

        if hasattr(viewpoint_cam, 'tool_mask'):# tissue mask
            tool_mask = viewpoint_cam.tool_mask.cuda().bool()
        else:
            tool_mask = torch.ones_like(gt_image[0:1]).bool()

        if hasattr(viewpoint_cam, 'original_sky_mask'):
            sky_mask = viewpoint_cam.original_sky_mask.cuda()
        else:
            sky_mask = None
            
        if hasattr(viewpoint_cam, 'original_obj_bound'):
            obj_bound = viewpoint_cam.original_obj_bound.cuda().bool()
        else:
            obj_bound = torch.zeros_like(gt_image[0:1]).bool()
        
        if (iteration - 1) == training_args.debug_from:
            cfg.render.debug = True
            
        print('!!!!TODO only support tisseu for now...')
        bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_pkg = render(viewpoint_cam, controller.tissue, cfg.render, background)
        image, depth, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], \
                render_pkg["visibility_filter"], render_pkg["radii"]
        acc = torch.zeros_like(depth)
        print('todo not sure acc...')

        scalar_dict = dict()

        from utils.loss_utils import l1_loss

        # tissue loss
        Ll1 = l1_loss(image, gt_image, mask)
        scalar_dict['l1_loss'] = Ll1.item()
        loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (1.0 - ssim(image.to(torch.double), gt_image.to(torch.double), mask=mask))
        print('Missing Depth loss...')

        # hard code tool_loss 
        hard_code_tool_loss = True
        hard_code_tool_loss = False
        if hard_code_tool_loss:
            Ll1_tool = l1_loss(image, gt_image, tool_mask)
            scalar_dict['l1_tool_loss'] = Ll1_tool.item()
            tool_loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1_tool \
                + optim_args.lambda_dssim * (1.0 - ssim(image.to(torch.double), gt_image.to(torch.double), \
                                                        mask=tool_mask))
            loss += tool_loss




        # sky loss
        if optim_args.lambda_sky > 0 and controller.include_sky and sky_mask is not None:
            assert 0, 'temp disabled '
            acc = torch.clamp(acc, min=1e-6, max=1.-1e-6)
            sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -torch.log(acc)).mean()
            if len(optim_args.lambda_sky_scale) > 0:
                sky_loss *= optim_args.lambda_sky_scale[viewpoint_cam.meta['cam']]
            scalar_dict['sky_loss'] = sky_loss.item()
            loss += optim_args.lambda_sky * sky_loss

        # semantic loss
        if optim_args.lambda_semantic > 0 and data_args.get('use_semantic', False) and 'semantic' in viewpoint_cam.meta:
            assert 0, 'temp disabled '
            gt_semantic = viewpoint_cam.meta['semantic'].cuda().long() # [1, H, W]
            if torch.all(gt_semantic == -1):
                semantic_loss = torch.zeros_like(Ll1)
            else:
                semantic = render_pkg['semantic'].unsqueeze(0) # [1, S, H, W]
                semantic_loss = torch.nn.functional.cross_entropy(
                    input=semantic, 
                    target=gt_semantic,
                    ignore_index=-1, 
                    reduction='mean'
                )
            scalar_dict['semantic_loss'] = semantic_loss.item()
            loss += optim_args.lambda_semantic * semantic_loss
        
        if optim_args.lambda_reg > 0 and controller.include_obj and iteration >= optim_args.densify_until_iter:
            assert 0, 'temp disabled '
            render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, controller)
            image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = torch.clamp(acc_obj, min=1e-6, max=1.-1e-6)
            # box_reg_loss = controller.get_box_reg_loss()
            # scalar_dict['box_reg_loss'] = box_reg_loss.item()
            # loss += optim_args.lambda_reg * box_reg_loss

            obj_acc_loss = torch.where(obj_bound, 
                -(acc_obj * torch.log(acc_obj) +  (1. - acc_obj) * torch.log(1. - acc_obj)), 
                -torch.log(1. - acc_obj)).mean()
            scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
            loss += optim_args.lambda_reg * obj_acc_loss
            # obj_acc_loss = -((acc_obj * torch.log(acc_obj) +  (1. - acc_obj) * torch.log(1. - acc_obj))).mean()
            # scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
            # loss += optim_args.lambda_reg * obj_acc_loss
        
        # lidar depth loss
        if optim_args.lambda_depth_lidar > 0 and 'lidar_depth' in viewpoint_cam.meta:   
            assert 0, 'temp disabled '
            lidar_depth = viewpoint_cam.meta['lidar_depth'].cuda() # [1, H, W]
            depth_mask = torch.logical_and((lidar_depth > 0.), mask)
            # depth_mask[obj_bound] = False
            if torch.nonzero(depth_mask).any():
                expected_depth = depth / (render_pkg['acc'] + 1e-10)  
                depth_error = torch.abs((expected_depth[depth_mask] - lidar_depth[depth_mask]))
                depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
                lidar_depth_loss = depth_error.mean()
                scalar_dict['lidar_depth_loss'] = lidar_depth_loss
            else:
                lidar_depth_loss = torch.zeros_like(Ll1)  
            loss += optim_args.lambda_depth_lidar * lidar_depth_loss
                    
        # color correction loss
        if optim_args.lambda_color_correction > 0 and controller.use_color_correction:
            assert 0, 'temp disabled '
            color_correction_reg_loss = controller.color_correction.regularization_loss(viewpoint_cam)
            scalar_dict['color_correction_reg_loss'] = color_correction_reg_loss.item()
            loss += optim_args.lambda_color_correction * color_correction_reg_loss
        
        # pose correction loss
        if optim_args.lambda_pose_correction > 0 and controller.use_pose_correction:
            assert 0, 'temp disabled '
            pose_correction_reg_loss = controller.pose_correction.regularization_loss()
            scalar_dict['pose_correction_reg_loss'] = pose_correction_reg_loss.item()
            loss += optim_args.lambda_pose_correction * pose_correction_reg_loss
                    
        # scale flatten loss
        if optim_args.lambda_scale_flatten > 0:
            assert 0, 'temp disabled '
            scale_flatten_loss = controller.background.scale_flatten_loss()
            scalar_dict['scale_flatten_loss'] = scale_flatten_loss.item()
            loss += optim_args.lambda_scale_flatten * scale_flatten_loss
        
        # opacity sparse loss
        if optim_args.lambda_opacity_sparse > 0:
            assert 0, 'temp disabled '
            opacity = controller.get_opacity
            opacity = opacity.clamp(1e-6, 1-1e-6)
            log_opacity = opacity * torch.log(opacity)
            log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
            sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[visibility_filter].mean()
            scalar_dict['opacity_sparse_loss'] = sparse_loss.item()
            loss += optim_args.lambda_opacity_sparse * sparse_loss
                
        # normal loss
        if optim_args.lambda_normal_mono > 0 and 'mono_normal' in viewpoint_cam.meta and 'normals' in render_pkg:
            assert 0, 'temp disabled '
            if sky_mask is None:
                normal_mask = mask
            else:
                normal_mask = torch.logical_and(mask, ~sky_mask)
                normal_mask = normal_mask.squeeze(0)
                normal_mask[:50] = False
                
            normal_gt = viewpoint_cam.meta['mono_normal'].permute(1, 2, 0).cuda() # [H, W, 3]
            R_c2w = viewpoint_cam.world_view_transform[:3, :3]
            normal_gt = torch.matmul(normal_gt, R_c2w.T) # to world space
            normal_pred = render_pkg['normals'].permute(1, 2, 0) # [H, W, 3]    
            
            normal_l1_loss = torch.abs(normal_pred[normal_mask] - normal_gt[normal_mask]).mean()
            normal_cos_loss = (1. - torch.sum(normal_pred[normal_mask] * normal_gt[normal_mask], dim=-1)).mean()
            scalar_dict['normal_l1_loss'] = normal_l1_loss.item()
            scalar_dict['normal_cos_loss'] = normal_cos_loss.item()
            normal_loss = normal_l1_loss + normal_cos_loss
            loss += optim_args.lambda_normal_mono * normal_loss
            
        scalar_dict['loss'] = loss.item()

        loss.backward()


        # follow deform3dgs
        # print('todo Correct? ')

        iter_end.record()
        is_save_images = True
        # if is_save_images and (iteration % 1000 == 0):
        if is_save_images and (iteration % 10 == 0):
            # row0: gt_image, image, depth
            # row1: acc, image_obj, acc_obj
            depth_colored, _ = visualize_depth_numpy(depth.detach().cpu().numpy().squeeze(0))
            depth_colored = depth_colored[..., [2, 1, 0]] / 255.
            depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float().cuda()
            row0 = torch.cat([gt_image, image, depth_colored], dim=2)
            acc = acc.repeat(3, 1, 1)
            with torch.no_grad():
                print('!!!!TODO only support tisseu for now...')
                bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                render_pkg_obj = render(viewpoint_cam, controller.tissue, cfg.render, background)
                image_obj, depth_obj = render_pkg_obj["render"], render_pkg_obj['depth']
                # render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians)
                # image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']

                depth_obj = depth_obj.repeat(3, 1, 1).to(image_obj.device) 
                place_holder = torch.zeros_like(depth_obj).to(depth_obj.device)
                row1 = torch.cat([image_obj, depth_obj, place_holder], dim=2)

            image_to_show = torch.cat([row0, row1], dim=1)
            image_to_show = torch.clamp(image_to_show, 0.0, 1.0)
            os.makedirs(f"{cfg.model_path}/log_images", exist_ok = True)
            save_img_torch(image_to_show, f"{cfg.model_path}/log_images/{iteration}.jpg")
        
        with torch.no_grad():
            # Log
            tensor_dict = dict()
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr(image, gt_image, mask).mean().float() + 0.6 * ema_psnr_for_log
            if viewpoint_cam.id not in psnr_dict:
                psnr_dict[viewpoint_cam.id] = psnr(image, gt_image, mask).mean().float()
            else:
                psnr_dict[viewpoint_cam.id] = 0.4 * psnr(image, gt_image, mask).mean().float() + 0.6 * psnr_dict[viewpoint_cam.id]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Exp": f"{cfg.task}-{cfg.exp_name}", 
                                          "Loss": f"{ema_loss_for_log:.{7}f},", 
                                          "PSNR": f"{ema_psnr_for_log:.{4}f}"})
                progress_bar.update(10)
            if iteration == training_args.iterations:
                progress_bar.close()
            # Log and save
            if (iteration in training_args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration,stage  ='')


            # Densification
            if iteration < optim_args.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                controller.set_visibility(include_list=list(set(controller.model_name_id.keys()) - set(['sky'])))
                controller.parse_camera(viewpoint_cam)  #update self.frame and other input for the rendering; cal the current #gs 
                controller.set_max_radii2D(radii, visibility_filter)
                controller.add_densification_stats(viewspace_point_tensor, visibility_filter)

                opacity_threshold = optim_args.opacity_threshold_fine_init - iteration*(optim_args.opacity_threshold_fine_init - optim_args.opacity_threshold_fine_after)/(optim_args.densify_until_iter)  
                densify_threshold = optim_args.densify_grad_threshold_fine_init - iteration*(optim_args.densify_grad_threshold_fine_init - optim_args.densify_grad_threshold_after)/(optim_args.densify_until_iter )  

                # densify and prune
                if iteration > optim_args.densify_from_iter and iteration % optim_args.densification_interval == 0 :
                    size_threshold = 20 if iteration > optim_args.opacity_reset_interval else None
                    controller.densify_and_prune(max_grad = densify_threshold, 
                                                    min_opacity = opacity_threshold, 
                                                exclude_list = [],
                                                extent = scene.cameras_extent, 
                                                max_screen_size = size_threshold,
                                                skip_prune = True)
                    
                if iteration > optim_args.pruning_from_iter and iteration % optim_args.pruning_interval == 0:
                    size_threshold = 40 if iteration > optim_args.opacity_reset_interval else None
                    controller.densify_and_prune(max_grad = densify_threshold, 
                                                    min_opacity = opacity_threshold, 
                                                exclude_list = [],
                                                extent = scene.cameras_extent, 
                                                max_screen_size = size_threshold,
                                                skip_densify = True)
                # reset opacity
                if iteration % optim_args.opacity_reset_interval == 0 or (data_args.white_background and iteration == optim_args.densify_from_iter):
                    print("reset opacity")
                    controller.reset_opacity()
            #training report happen here?
            
            
            # Optimizer step
            # if iteration < optim_args.iterations:
            if iteration < training_args.iterations:
                controller.update_optimizer()

                # controller.optimizer.step()
                # controller.optimizer.zero_grad(set_to_none = True)





            # ////////////////////////////////////////////


            if render_stree_param_for_ori_train_report!= None:
                print('todo ugly')
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                            training_args.test_iterations, scene, render, [render_stree_param_for_ori_train_report, background])

            # Optimizer step
            if iteration < training_args.iterations:
                controller.update_optimizer()
            # assert 0,training_args.checkpoint_iterations
            if (iteration in training_args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                state_dict = controller.save_state_dict(is_final=(iteration == training_args.iterations))
                state_dict['iter'] = iteration
                ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{iteration}.pth')
                print('save state dict in',ckpt_path)
                torch.save(state_dict, ckpt_path)

from render_misgs import MisGaussianRenderer
def training_report_misgs(tb_writer, iteration, scalar_stats, tensor_stats, testing_iterations, scene: Scene, 
                             renderer: MisGaussianRenderer):
    if tb_writer:
        try:
            for key, value in scalar_stats.items():
                tb_writer.add_scalar('train/' + key, value, iteration)
            for key, value in tensor_stats.items():
                tb_writer.add_histogram('train/' + key, value, iteration)
        except:
            print('Failed to write to tensorboard')
            
            
    # Report test and samples of training set
    if iteration in testing_iterations:
    # if iteration in testing_iterations or True:
        print('todo hard set the flag to True log in test')
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test/test_view', 
                               'cameras' : scene.getTestCameras()},
                              {'name': 'test/train_view', 
                               'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderer.render(viewpoint, scene.gaussians_or_controller)["rgb"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    if hasattr(viewpoint, 'original_mask'):
                        mask = viewpoint.original_mask.cuda().bool()
                    else:
                        mask = torch.ones_like(gt_image[0]).bool()
                    from utils.loss_utils import l1_loss
                    l1_test += l1_loss(image, gt_image, mask).mean().double()
                    psnr_test += psnr(image, gt_image, mask).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("test/opacity_histogram", scene.gaussians_or_controller.get_opacity, iteration)
            tb_writer.add_scalar('test/points_total', scene.gaussians_or_controller.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

 