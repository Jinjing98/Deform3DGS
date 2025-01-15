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
from gaussian_renderer import render_flow as fdm_render

import sys
from scene import  Scene
from scene.flexible_deform_model import TissueGaussianModel
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
def training_misgsmodel(args,use_streetgs_render = False,eval_n_log_test_cam =False):
    #///////////////////////////////////////
    #hard code
    renderOnce = False
    renderOnce = True  
    compo_all_gs_ordered_renderonce=['tissue','obj_tool1']
    # compo_all_gs_ordered_renderonce=['tissue']
    # compo_all_gs_ordered_renderonce=['obj_tool1']

    # log tb
    # use_ema_train = True # recommeded
    use_ema_train = False # has its advantage
    use_ema_test = False # the only choice for test
    
    other_param_dict = None
    dbg_print = True
    dbg_print = False
    remain_redundant = True
    #///////////////////////////////////////

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
    from train import prepare_output_and_logger
    tb_writer = prepare_output_and_logger(model_path=cfg.expname, write_args=args)
    timer = Timer()
    load_other_obj_meta=True #load within the sceneinfo
    load_pcd_dict_in_sceneinfo=True #piece wise pcd init
    print('////////////////***************///////////')
    print('MisGS reuse the Scene function of deform3dgs (only break it down)')
    scene = Scene(mod_stree_param,
                  load_other_obj_meta=load_other_obj_meta,
                  new_cfg=cfg,
                  load_pcd_dict_in_sceneinfo=load_pcd_dict_in_sceneinfo,
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
                                eval_n_log_test_cam = eval_n_log_test_cam,  
                                renderOnce=renderOnce,
                                compo_all_gs_ordered_renderonce=compo_all_gs_ordered_renderonce,
                                use_ema_train=use_ema_train,
                                use_ema_test=use_ema_test,
                               )



def compute_more_metrics(gt_image,
                         renderOnce,image_all,cfg,
                         tissue_mask,tool_mask,more_to_log,
                            use_ema = True,
                            ema_psnr_for_log_tissue = None,
                            ema_psnr_for_log_tool = None,
                            dir_append = ''
                            ):
    # tensor_dict = dict()
    # Progress bar
    # ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
    # Log PSNR in tb
    # psnr_weight_ori = 0 # set to 0,then it would be compariable to the one computed in deform3dgs
    psnr_weight_ori = 0.6 if use_ema else 0# set to 0,then it would be compariable to the one computed in deform3dgs
    psnr_weight_current = 1-psnr_weight_ori
    log_psnr_name = 'ema_psnr' if use_ema else 'crt_psnr'
    if use_ema:
        assert ema_psnr_for_log_tissue!= None
        assert ema_psnr_for_log_tool!= None

    if renderOnce:
        image_tissue = image_all
        image_tool = image_all
    if cfg.model.nsg.include_tissue:
        # exponential moving average
        ema_psnr_for_log_tissue = psnr_weight_current * psnr(image_tissue, gt_image, tissue_mask).mean().float() 
        + psnr_weight_ori * ema_psnr_for_log_tissue
        # if viewpoint_cam.id not in psnr_dict_tissue:
        #     psnr_dict_tissue[viewpoint_cam.id] = psnr(image_tissue, gt_image, tissue_mask).mean().float()
        # else:
        #     psnr_dict_tissue[viewpoint_cam.id] = psnr_weight_current * psnr(image_tissue, gt_image, tissue_mask).mean().float() 
        #     + psnr_weight_ori * psnr_dict_tissue[viewpoint_cam.id]
        more_to_log[f'tissue/{log_psnr_name}{dir_append}'] = ema_psnr_for_log_tissue
    else:
        assert 0,'alwasy include tissue'

    if cfg.model.nsg.include_obj:
        # exponential moving average
        ema_psnr_for_log_tool = psnr_weight_current * psnr(image_tool, gt_image, tool_mask).mean().float() 
        + psnr_weight_ori * ema_psnr_for_log_tool
        # if viewpoint_cam.id not in psnr_dict_tool:
        #     psnr_dict_tool[viewpoint_cam.id] = psnr(image_tool, gt_image, tool_mask).mean().float()
        # else:
        #     psnr_dict_tool[viewpoint_cam.id] = psnr_weight_current * psnr(image_tool, gt_image, tool_mask).mean().float() 
        #     + psnr_weight_ori * psnr_dict_tool[viewpoint_cam.id]
        more_to_log[f'tool/{log_psnr_name}{dir_append}'] =  ema_psnr_for_log_tool
    else:
        assert 0,'alwasy include tool'
    return ema_psnr_for_log_tissue,ema_psnr_for_log_tool,more_to_log

# 
def render_misgs_n_compute_loss(controller,viewpoint_cam,cfg,training_args,optim_args,
                                renderOnce,compo_all_gs_ordered_renderonce,
                                debug_getxyz_misgs,
                                iteration,
                                skip_loss_compute = False):
    # viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    gt_image = viewpoint_cam.original_image.cuda()
    gt_depth = viewpoint_cam.original_depth.cuda()
    if hasattr(viewpoint_cam, 'tissue_mask'):# tissue mask
        tissue_mask = viewpoint_cam.tissue_mask.cuda().bool()
    else:
        tissue_mask = torch.ones_like(gt_image[0:1]).bool()
    if hasattr(viewpoint_cam, 'tool_mask'):# tissue mask
        tool_mask = viewpoint_cam.tool_mask.cuda().bool()
    else:
        tool_mask = torch.ones_like(gt_image[0:1]).bool()
    if (iteration - 1) == training_args.debug_from:
        cfg.render.debug = True

    bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    scalar_dict = dict()
    from utils.loss_utils import l1_loss
    radii_all_compo_adc = {}
    visibility_filters_all_compo_adc = {}
    viewspace_point_tensors_all_compo_adcdict = {}
    model_names_all_compo_adc = []

    if renderOnce:            
        compo_all_gs_ordered = compo_all_gs_ordered_renderonce
        for name in compo_all_gs_ordered:
            if 'tissue' in name:
                assert cfg.model.nsg.include_tissue, name
            elif 'tool' in name:
                assert cfg.model.nsg.include_obj, name
            else:
                assert 0,name 
        render_pkg_all,compo_all_gs_ordered_idx = fdm_render(viewpoint_cam, 
                                        None, 
                                        cfg.render, 
                                        background,
                                    debug_getxyz_misgs=debug_getxyz_misgs,
                                    misgs_model=controller,
                                    single_compo_or_list=compo_all_gs_ordered,
                                    tool_parse_cam_again = True,#1st time
                                    )
        # get the dict below     
        image_all = render_pkg_all["render"]
        depth_all = render_pkg_all["depth"]
        if not skip_loss_compute:
            Ll1 = l1_loss(image_all, gt_image, tissue_mask)
            #scalar_dict['l1_loss'] = Ll1.item()
            loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + \
                optim_args.lambda_dssim * (1.0 - ssim(image_all.to(torch.double), \
                                                    gt_image.to(torch.double), mask=tissue_mask))
            
            Ll1_tool = l1_loss(image_all, gt_image, tool_mask)
            #scalar_dict['l1_tool_loss'] = Ll1_tool.item()
            tool_loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1_tool \
                + optim_args.lambda_dssim * (1.0 - ssim(image_all.to(torch.double), gt_image.to(torch.double), \
                                                        mask=tool_mask))
            # more recommended than edit order list
            loss += tool_loss
            # loss = 0*loss+tool_loss
            # loss = loss+tool_loss*0


            #//////////////////////////////////
            # remain tissue depth supervsion
            use_tissue_depth = True
            use_tool_depth = True
            # seems not much difference?
            # use_tissue_depth = False
            # use_tool_depth = False
            depth_all = depth_all.unsqueeze(0)
            gt_depth = gt_depth.unsqueeze(0)
            if (gt_depth!=0).sum() < 10:
                assert 0
                depth_loss = torch.tensor(0.).cuda()
            else:
                # inverse depth before compute loss
                depth_all[depth_all!=0] = 1 / depth_all[depth_all!=0]
                gt_depth[gt_depth!=0] = 1 / gt_depth[gt_depth!=0]
                depth_loss = torch.tensor(0.).cuda()
                if use_tissue_depth:
                    depth_loss += l1_loss(depth_all, gt_depth, tissue_mask)
                if use_tool_depth:
                    depth_loss += l1_loss(depth_all, gt_depth, tool_mask)
            loss += depth_loss
            # loss = depth_loss
            #//////////////////////////////////



        for model_name,(start_idx,end_idx) in compo_all_gs_ordered_idx.items():
                model_names_all_compo_adc.append(model_name)
                assert end_idx<len(render_pkg_all["radii"]),f'end_idx {end_idx} should be less than {len(render_pkg_all["radii"])}'
                assert len(render_pkg_all["radii"])==len(render_pkg_all["visibility_filter"])
                assert len(render_pkg_all["radii"])==len(render_pkg_all["viewspace_points"])
                radii_all_compo_adc[model_name] = render_pkg_all["radii"][start_idx:(end_idx+1)]
                visibility_filters_all_compo_adc[model_name] = render_pkg_all["visibility_filter"][start_idx:(end_idx+1)]
                # viewspace_point_tensors_all_compo_adcdict[model_name] = render_pkg_all["viewspace_points"][start_idx:(end_idx+1)]
        assert model_names_all_compo_adc == compo_all_gs_ordered,'sanity check'

    
    else:
        if cfg.model.nsg.include_tissue:
            render_pkg_tissue,_ = fdm_render(viewpoint_cam, controller.tissue, cfg.render, background,
                                            single_compo_or_list='tissue',
                                            tool_parse_cam_again = False,# no need for tissue
                                            )
            image_tissue, depth_tissue, viewspace_point_tensor_tissue, visibility_filter_tissue, radii_tissue = \
                render_pkg_tissue["render"], render_pkg_tissue["depth"], render_pkg_tissue["viewspace_points"], \
                    render_pkg_tissue["visibility_filter"], render_pkg_tissue["radii"]
            acc_tissue = torch.zeros_like(depth_tissue)
            if not skip_loss_compute:
                Ll1 = l1_loss(image_tissue, gt_image, tissue_mask)
                scalar_dict['l1_loss'] = Ll1.item()
                loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + \
                    optim_args.lambda_dssim * (1.0 - ssim(image_tissue.to(torch.double), \
                                                        gt_image.to(torch.double), mask=tissue_mask))
                # print('Missing Depth loss...')
            
            # register for adc
            model_names_all_compo_adc.append('tissue')
            radii_all_compo_adc['tissue'] = radii_tissue
            visibility_filters_all_compo_adc['tissue'] = visibility_filter_tissue
            viewspace_point_tensors_all_compo_adcdict['tissue'] = viewspace_point_tensor_tissue
            
            
        else:
            assert 0,'alwasy include tissue'

        if cfg.model.nsg.include_obj:
            # render_pkg_tool = gaussians_renderer.render_object(viewpoint_cam, gaussians)
            render_pkg_tool,_ = fdm_render(viewpoint_cam, controller.obj_tool1, cfg.render, background,
                                        debug_getxyz_misgs=debug_getxyz_misgs,
                                        misgs_model=controller,
                                        single_compo_or_list='tool',
                                        tool_parse_cam_again = True,#1st time
                                        )
            image_tool, depth_tool, viewspace_point_tensor_tool, visibility_filter_tool, radii_tool = \
                render_pkg_tool["render"], render_pkg_tool["depth"], render_pkg_tool["viewspace_points"], \
                    render_pkg_tool["visibility_filter"], render_pkg_tool["radii"]
            if not skip_loss_compute:
                Ll1_tool = l1_loss(image_tool, gt_image, tool_mask)
                scalar_dict['l1_tool_loss'] = Ll1_tool.item()
                tool_loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1_tool \
                    + optim_args.lambda_dssim * (1.0 - ssim(image_tool.to(torch.double), gt_image.to(torch.double), \
                                                            mask=tool_mask))
                # loss += tool_loss
                # loss = 0*loss+tool_loss
                loss = loss+tool_loss*0

            # register for adc
            model_names_all_compo_adc.append('obj_tool1')
            radii_all_compo_adc['obj_tool1'] = radii_tool
            visibility_filters_all_compo_adc['obj_tool1'] = visibility_filter_tool
            viewspace_point_tensors_all_compo_adcdict['obj_tool1'] = viewspace_point_tensor_tool
    
    if skip_loss_compute:
         Ll1, loss = None,None

    return render_pkg_all,compo_all_gs_ordered_idx, Ll1, loss,scalar_dict,\
        gt_image,image_all,tissue_mask,tool_mask,\
        radii_all_compo_adc,visibility_filters_all_compo_adc,model_names_all_compo_adc,\
            viewspace_point_tensors_all_compo_adcdict,visibility_filters_all_compo_adc,\
                model_names_all_compo_adc




def scene_reconstruction_misgs(cfg, controller, scene, tb_writer,
                               render_stree_param_for_ori_train_report = None,
                               use_streetgs_render = False,
                               debug_getxyz_misgs = True,
                            #    debug_getxyz_misgs = False,
                               eval_n_log_test_cam = False,
                                renderOnce = True,
                                compo_all_gs_ordered_renderonce=['tissue','obj_tool1'],
                                # use_ema_train = True # recommeded,
                                use_ema_train = False,
                                use_ema_test = False, # the only choice for test
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

    # from render_misgs import MisGaussianRenderer
    # gaussians_renderer = MisGaussianRenderer(cfg=cfg)
    # from gaussian_renderer import render_flow as fdm_render
    # from gaussian_renderer.tool_renderer import tool_render

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    #streetgs traing added
    # ema_loss_for_log = 0.0
    ema_psnr_for_log_tissue_trn = 0.0
    ema_psnr_for_log_tool_trn = 0.0
    ema_psnr_for_log_tissue_test = 0.0
    ema_psnr_for_log_tool_test = 0.0

    # psnr_dict_tissue = {}
    # psnr_dict_tool = {}
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

        # print('potentially moving out the loop for effeciency.')
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        # print("Loading Training Cameras")
        # self.train_camera = scene_info.train_cameras 

        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg_all,compo_all_gs_ordered_idx,  Ll1, loss, scalar_dict,\
            gt_image,image_all,tissue_mask,tool_mask,\
                radii_all_compo_adc,visibility_filters_all_compo_adc,model_names_all_compo_adc,\
                    viewspace_point_tensors_all_compo_adcdict,visibility_filters_all_compo_adc,\
                        model_names_all_compo_adc = render_misgs_n_compute_loss(controller,viewpoint_cam,cfg,training_args,optim_args,
                                renderOnce,compo_all_gs_ordered_renderonce,
                                debug_getxyz_misgs,
                                iteration,
                            )
        # controller,viewpoint_cam,cfg,training_args,optim_args,
        #                         renderOnce,compo_all_gs_ordered_renderonce,
        #                         debug_getxyz_misgs,
        #                         iteration,
        #                         skip_loss_compute = False

        scalar_dict['loss'] = loss.item()
        loss.backward()
        iter_end.record()

        more_to_log = {}
        # log psnr for training

        with torch.no_grad():
            # maintain more_to_log
            # udpate ema_psnr_for_log_tool ema_psnr_for_log_tissue
            ema_psnr_for_log_tissue_trn,ema_psnr_for_log_tool_trn,more_to_log = \
                compute_more_metrics(gt_image,renderOnce,image_all,cfg,tissue_mask,tool_mask,more_to_log,
                                     use_ema=use_ema_train,
                                     ema_psnr_for_log_tissue=ema_psnr_for_log_tissue_trn,
                                     ema_psnr_for_log_tool=ema_psnr_for_log_tool_trn,
                                     dir_append = '',
                                     )
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Exp": f"{cfg.task}-{cfg.exp_name}", 
                                        #   "Loss": f"{ema_loss_for_log:.{7}f},", 
                                          "PSNR_tissue": f"{ema_psnr_for_log_tissue_trn:.{4}f}",
                                          "PSNR_tool": f"{ema_psnr_for_log_tool_trn:.{4}f}",
                                          }
                                          )
                progress_bar.update(10)
            if iteration == training_args.iterations:
                progress_bar.close()
            # Log and save
            if (iteration in training_args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                stage =''
                scene.save(iteration,stage =stage)
                if isinstance(scene.gaussians_or_controller, MisGaussianModel):
                    pose_model_root = os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration))
                    pose_model_path = os.path.join(pose_model_root, "pose_model.pth")
                    if scene.gaussians_or_controller.poses_all_objs is not None:
                        state_dict = scene.gaussians_or_controller.poses_all_objs.save_state_dict()
                        torch.save(state_dict, pose_model_path)

        # log psnr for test
        with torch.no_grad():
            if eval_n_log_test_cam:
                #render first
                test_viewpoint_stack = scene.getTestCameras().copy()
                test_viewpoint_cam: Camera = test_viewpoint_stack.pop(randint(0, len(test_viewpoint_stack) - 1))

                test_render_pkg_all,test_compo_all_gs_ordered_idx,  test_Ll1, test_loss, test_scalar_dict,\
                    test_gt_image,test_image_all,test_tissue_mask,test_tool_mask,\
                        test_radii_all_compo_adc,test_visibility_filters_all_compo_adc,test_model_names_all_compo_adc,\
                            test_viewspace_point_tensors_all_compo_adcdict,test_visibility_filters_all_compo_adc,\
                                test_model_names_all_compo_adc = render_misgs_n_compute_loss(controller,test_viewpoint_cam,cfg,training_args,optim_args,
                                        renderOnce,compo_all_gs_ordered_renderonce,
                                        debug_getxyz_misgs,
                                        iteration,
                                        skip_loss_compute=True,
                                    )
                # compute metric
                # maintain more_to_log
                ema_psnr_for_log_tissue_test,ema_psnr_for_log_tool_test,more_to_log = \
                    compute_more_metrics(test_gt_image,renderOnce,test_image_all,cfg,test_tissue_mask,test_tool_mask,more_to_log,
                                        use_ema=use_ema_test,
                                        ema_psnr_for_log_tissue=ema_psnr_for_log_tissue_test,
                                        ema_psnr_for_log_tool=ema_psnr_for_log_tool_test,
                                        dir_append = '_test'
                                        )
                




        with torch.no_grad():
            # todo:xyz_gradient_accum
            # problem lies in fusing set_max_radii2D_all_models and add_densification_stats_all_models of misgs model with its own function
            # also check out where else can leverage the idx dict properly
            # Densification
            if iteration < optim_args.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                controller.set_visibility(include_list=list(set(controller.model_name_id.keys()) ))
                # newly updated streetgs remove this...
                # print('todo: check if the below line is necessary here?why i need it in densify?' )
                # controller.parse_camera(viewpoint_cam,skip_obj_pose = True)  #update self.frame and other input for the rendering; cal the current #gs 
                controller.set_max_radii2D_all_models(radiis = radii_all_compo_adc, 
                                                      visibility_filters = visibility_filters_all_compo_adc,
                                                      model_names = model_names_all_compo_adc)
                # it is imporatant to parse viewspace_point_tensors as one torch tensor rather dict when render_once, or will loss the saved grad--but why?
                controller.add_densification_stats_all_models(viewspace_point_tensors = viewspace_point_tensors_all_compo_adcdict if not renderOnce else render_pkg_all['viewspace_points'], 
                                                              visibility_filters = visibility_filters_all_compo_adc,
                                                              model_names = model_names_all_compo_adc,
                                                              compo_all_gs_ordered_idx = compo_all_gs_ordered_idx if renderOnce else None,
                                                              )

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
        with torch.no_grad():
            if render_stree_param_for_ori_train_report!= None:
                training_report(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end), 
                            training_args.test_iterations, scene, 
                            more_to_log = more_to_log,
                            # render,[render_stree_param_for_ori_train_report, background]
                            )

            # Optimizer step
            if iteration < training_args.iterations:
                controller.update_optimizer()
            # assert 0,training_args.checkpoint_iterations
            if (iteration in training_args.checkpoint_iterations):
                #not used in deform3dgs
                pass
                # print("\n[ITER {}] Saving Checkpoint".format(iteration))
                # state_dict = controller.save_state_dict(is_final=(iteration == training_args.iterations))
                # state_dict['iter'] = iteration
                # ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{iteration}.pth')
                # print('save state dict in',ckpt_path)
                # torch.save(state_dict, ckpt_path)




        is_save_images = True
        # if is_save_images and (iteration % 1000 == 0):
        if is_save_images \
            and (iteration % 10 == 0)\
            and iteration > int(0.9*training_args.iterations)\
                :
            # row0: gt_image, image, depth
            # row1: acc, image_obj, acc_obj
            # depth_colored, _ = visualize_depth_numpy(depth.detach().cpu().numpy().squeeze(0))
            # depth_colored = depth_colored[..., [2, 1, 0]] / 255.
            # depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float().cuda()
            # row0 = torch.cat([gt_image, image, depth_colored], dim=2)
            row0 = torch.cat([gt_image, gt_image,gt_image], dim=2)
            # acc = acc.repeat(3, 1, 1)
            with torch.no_grad():
                bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                image_to_show_list = [row0]
                # if controller.obj_tool:
                for model_name in controller.model_name_id.keys():
                    if controller.get_visibility(model_name=model_name):
                        sub_gs_model = getattr(controller, model_name)
                        try:
                            assert 'tissue' in model_name
                            render_pkg,_= fdm_render(viewpoint_cam, sub_gs_model, cfg.render, background,
                                                     single_compo_or_list='tissue',
                                                     tool_parse_cam_again = False,#no need for tissue 
                                                     )
                        except:
                            assert 'tool' in model_name
                            render_pkg,_= fdm_render(viewpoint_cam, sub_gs_model, cfg.render, background,
                                                    debug_getxyz_misgs = debug_getxyz_misgs,
                                                    misgs_model = controller,
                                                    single_compo_or_list='tool',
                                                    tool_parse_cam_again = False,#no need again 
                                                    )

                        image_obj, depth_obj = render_pkg["render"], render_pkg['depth']

                        depth_obj = depth_obj.repeat(3, 1, 1).to(image_obj.device) 
                        place_holder = torch.zeros_like(depth_obj).to(depth_obj.device)
                        row_i = torch.cat([image_obj, depth_obj, place_holder], dim=2)
                        image_to_show_list.append(row_i)

            # image_to_show = torch.cat([row0, row1], dim=1)
            image_to_show = torch.cat(image_to_show_list, dim=1)
            image_to_show = torch.clamp(image_to_show, 0.0, 1.0)
            os.makedirs(f"{cfg.model_path}/log_images", exist_ok = True)
            # save_img_torch(image_to_show, f"{cfg.model_path}/log_images/{iteration}.jpg")
            # log_img_name = f'it{iteration}_name{viewpoint_cam.image_name}_id{viewpoint_cam.id}_time{viewpoint_cam.time}'
            log_img_name = f'id{viewpoint_cam.id}_it{iteration}_name{viewpoint_cam.image_name}_time{viewpoint_cam.time:.2f}'
            save_img_torch(image_to_show, f"{cfg.model_path}/log_images/{log_img_name}.jpg")

        




# from render_misgs import MisGaussianRenderer
from gaussian_renderer.misgs_renderer import MisGaussianRenderer
def training_report_misgs(tb_writer, 
                          iteration, 
                          scalar_stats, 
                          tensor_stats, 
                          testing_iterations, 
                          scene: Scene,
                          renderer: MisGaussianRenderer,
                          cfg = None,
                          ):
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
                    
                    # try to get all the masks
                    if hasattr(viewpoint, 'tissue_mask'):
                        tissue_mask = viewpoint.tissue_mask.cuda().bool()
                    else:
                        tissue_mask = torch.ones_like(gt_image[0]).bool()
                    
                    # self.nsg.include_obj = False #True # include object
                    # self.nsg.opt_track = False # tracklets optimization
                    # always have tissue in the background
                    if cfg.model.nsg.include_tissue:
                        from utils.loss_utils import l1_loss
                        l1_test += l1_loss(image, gt_image, tissue_mask).mean().double()
                        psnr_test += psnr(image, gt_image, tissue_mask).mean().double()
                    else:
                        assert 0,'always include tissue'


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

 

         # if renderOnce:
        #     #need to be 32738*3
        #     print(f'??debug fuse: render_pkg_all viewspace_points grad {render_pkg_all["viewspace_points"].grad}')
        #     assert render_pkg_all['viewspace_points'].grad!=None 
        # else:
        #     #30721*3 2017*3
        #     try:
        #         print(f"*************{render_pkg_tissue['viewspace_points'].grad.shape} {render_pkg_tool['viewspace_points'].grad.shape}")
        #     except:
        #         pass
        # # follow deform3dgs