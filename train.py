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
from utils.loss_utils import l1_loss
from gaussian_renderer import render_flow as fdm_render

import sys
from scene import  Scene
from scene.flexible_deform_model import TissueGaussianModel
from scene.tool_model import ToolModel
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

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, train_iter, timer):
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None

    
    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    
    # lpips_model = lpips.LPIPS(net="vgg").cuda()
    video_cams = scene.getVideoCameras()
    
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()
        
    # ema_psnr_for_log = 0
    ema_psnr_for_log_tissue = 0
    ema_psnr_for_log_tool = 0
    for iteration in range(first_iter, final_iter+1):        

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
        # if iteration % 2 == 0:
            gaussians.oneupSHdegree()
            # assert 0, gaussians.active_sh_degree
        idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cams = [viewpoint_stack[idx]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []
        masks_tissue_dbg = []
        
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        
        for viewpoint_cam in viewpoint_cams:
            render_pkg,_ = fdm_render(viewpoint_cam, gaussians, pipe, background,
                                single_compo_or_list='tissue')
            image, depth, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda().float()
            gt_depth = viewpoint_cam.original_depth.cuda().float()
            mask = viewpoint_cam.mask.cuda()
            mask_tissue_dbg = viewpoint_cam.raw_tissue_mask.cuda()
            images.append(image.unsqueeze(0))
            depths.append(depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            masks_tissue_dbg.append(mask_tissue_dbg.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        depth_tensor = torch.cat(depths, 0)
        gt_image_tensor = torch.cat(gt_images,0)
        gt_depth_tensor = torch.cat(gt_depths, 0)
        mask_tensor = torch.cat(masks, 0)
        mask_tissue_dbg_tensor = torch.cat(masks_tissue_dbg, 0)
        
        more_to_log = {}

        Ll1 = l1_loss(image_tensor, gt_image_tensor, mask_tensor)
        
        if (gt_depth_tensor!=0).sum() < 10:
            depth_loss = torch.tensor(0.).cuda()
        else:
            depth_tensor[depth_tensor!=0] = 1 / depth_tensor[depth_tensor!=0]
            gt_depth_tensor[gt_depth_tensor!=0] = 1 / gt_depth_tensor[gt_depth_tensor!=0]
     
            depth_loss = l1_loss(depth_tensor, gt_depth_tensor, mask_tensor)
        
        #///////////////////////////////////////////////////
        if dataset.tool_mask == 'use':
            tissue_mask_tensor = mask_tensor
            tool_mask_tensor = ~mask_tensor
        elif dataset.tool_mask == 'inverse':
            tool_mask_tensor = mask_tensor
            tissue_mask_tensor = ~mask_tensor
        elif dataset.tool_mask == 'nouse':  
            # assert 0, 'todo try to use the actual mask saved in camera'
            # tool_mask_tensor = torch.zeros_like(mask_tensor)
            tissue_mask_tensor = mask_tissue_dbg_tensor #mask_tensor
            tool_mask_tensor = ~mask_tissue_dbg_tensor
        else:
            assert 0,dataset.tool_mask
        # Log PSNR in tb
        psnr_weight_ori = 0 # set to 0,then it would be compariable to the one computed in deform3dgs
        psnr_weight_ori = 0.6 # set to 0,then it would be compariable to the one computed in deform3dgs
        psnr_weight_current = 1-psnr_weight_ori
        log_psnr_name = 'ema_psnr' if psnr_weight_ori != 0 else 'crt_psnr'

        # psnr_ = psnr(image_tensor, gt_image_tensor, mask_tensor).mean().double()
        ema_psnr_for_log_tissue = psnr_weight_current * psnr(image_tensor, gt_image_tensor, 
                                                      tissue_mask_tensor).mean().double()
        + psnr_weight_ori * ema_psnr_for_log_tissue

        ema_psnr_for_log_tool = psnr_weight_current * psnr(image_tensor, gt_image_tensor, 
                                                      tool_mask_tensor).mean().double()
        + psnr_weight_ori * ema_psnr_for_log_tool

 
        assert isinstance(scene.gaussians_or_controller, TissueGaussianModel)
        more_to_log[f'tissue/{log_psnr_name}'] = ema_psnr_for_log_tissue
        more_to_log[f'tool/{log_psnr_name}'] = ema_psnr_for_log_tool
        #///////////////////////////////////////////////////



        loss = Ll1 + depth_loss 
        
        loss.backward()
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}",
                                          f"psnr_tissue": f"{ema_psnr_for_log_tissue:.{2}f}",
                                          f"psnr_tool": f"{ema_psnr_for_log_tool:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, 
                            # render, [pipe, background],
                            more_to_log=more_to_log,
                            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, 'fine')
            timer.start()
            
            # #//////////////////////////////////////
            # # Densification-test the wrarped densify+prune+reset_opacity of TissueGS model
            # if iteration < opt.densify_until_iter :
            #     gaussians.densify_and_prune_v0(iteration = iteration, 
            #                                 opt = opt, 
            #                                 cameras_extent = scene.cameras_extent,
            #                                 white_background = dataset.white_background,
            #                                 visibility_filter = visibility_filter,
            #                                 viewspace_point_tensor_grad = viewspace_point_tensor_grad,
            #                                 radii = radii)
            
            #//////////////////////////////////

            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
  
                opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                # densify and prune
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify_and_prune(max_grad = densify_threshold, 
                                                min_opacity = opacity_threshold, 
                                                extent = scene.cameras_extent, 
                                                max_screen_size = size_threshold,
                                                skip_prune = True)
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(max_grad = densify_threshold, 
                                                min_opacity = opacity_threshold, 
                                                extent = scene.cameras_extent, 
                                                max_screen_size = size_threshold,
                                                skip_densify = True)
                # reset opacity
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            # #//////////////////////////////////////
            # # Densification
            # if iteration < opt.densify_until_iter :
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

  
            #     opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
            #     densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
            #     if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
            #         size_threshold = 40 if iteration > opt.opacity_reset_interval else None
            #         gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
            #     # #from deformable 3d gs---surg-gs(0.0003/0.0002)/deformable3dgs(0.0007)
            #     # if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
            #     #     size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
            #     #     self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         print("reset opacity")
            #         gaussians.reset_opacity()   
            #          
            # # Optimizer step
            # if iteration < opt.iterations:
            #     gaussians.optimizer.step()
            #     gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, extra_mark,args = None):
    assert expname == args.model_path, f'{expname} {args.model_path}'
    tb_writer = prepare_output_and_logger(model_path=expname, write_args=args)
    gaussians = TissueGaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path# dataset save model param
    timer = Timer()
    # scene = Scene(dataset, gaussians)
    # convert 1 to 2 steps
    scene = Scene(dataset)
    scene.gs_init(gaussians_or_controller=gaussians,
                  reset_camera_extent=dataset.camera_extent)
    timer.start()
    #actual data loading
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, opt.iterations,timer)

def prepare_output_and_logger(model_path,write_args = None):  
    # if write_args.disable_tb=='Y':
    #     return None

    if not model_path:
        assert 0, model_path
    print("Output folder: {}".format(model_path))
    os.makedirs(model_path, exist_ok = True)
    with open(os.path.join(model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(write_args))))
    tb_writer = None
    
    if write_args.disable_tb=='Y':
        return None


    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, elapsed, testing_iterations, scene : Scene, 
                    # renderFunc, renderArgs,
                    more_to_log = {},
                    ):
    
    if tb_writer:
        tb_writer.add_scalar(f'loss/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'loss/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'other/iter_time', elapsed, iteration)

        #jj    
        from scene.mis_gaussian_model import MisGaussianModel
        if isinstance(scene.gaussians_or_controller, MisGaussianModel):
            # assert 0, scene.gaussians_or_controller.model
            tb_writer.add_scalar('other/total_points_tissue', scene.gaussians_or_controller.tissue.get_xyz.shape[0], iteration)
            tb_writer.add_scalar('other/total_points_tool', scene.gaussians_or_controller.obj_tool1.get_xyz.shape[0], iteration)
            tb_writer.add_scalar('other/total_points_all', scene.gaussians_or_controller.tissue.get_xyz.shape[0]
                                 +scene.gaussians_or_controller.obj_tool1.get_xyz.shape[0], iteration)
        else:
            if isinstance(scene.gaussians_or_controller, TissueGaussianModel):
                tgt = 'tissue'
            elif isinstance(scene.gaussians_or_controller, ToolModel):
                tgt = 'tool'
            else:
                assert 0,scene.gaussians_or_controller
            tb_writer.add_scalar(f'other/total_points_{tgt}', scene.gaussians_or_controller.get_xyz.shape[0], iteration)

        if more_to_log != {}:
            for k,v in more_to_log.items():
                tb_writer.add_scalar(f'{k}', v, iteration)


        # assert 0,  isinstance(scene.gaussians_or_controller, MisGaussianModel)
        # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians_or_controller.get_opacity, iteration)
        # torch.cuda.empty_cache()



def training_report_v0(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, 
                    # renderFunc, renderArgs,
                    ):
    
    if tb_writer:
        tb_writer.add_scalar(f'train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'iter_time', elapsed, iteration)

        #jj    
        from scene.mis_gaussian_model import MisGaussianModel
        if isinstance(scene.gaussians_or_controller, MisGaussianModel):
            # assert 0, scene.gaussians_or_controller.model
            tb_writer.add_scalar('total_points', scene.gaussians_or_controller.tissue.get_xyz.shape[0], iteration)
            tb_writer.add_scalar('total_points_tool', scene.gaussians_or_controller.obj_tool1.get_xyz.shape[0], iteration)
            tb_writer.add_scalar('total_points_all', scene.gaussians_or_controller.tissue.get_xyz.shape[0]+scene.gaussians_or_controller.obj_tool1.get_xyz.shape[0], iteration)
        else:
            tb_writer.add_scalar('total_points', scene.gaussians_or_controller.get_xyz.shape[0], iteration)

        # assert 0,  isinstance(scene.gaussians_or_controller, MisGaussianModel)
        # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians_or_controller.get_opacity, iteration)
        # torch.cuda.empty_cache()



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    
    
    # import os
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()

    #misgs hard code
    eval_n_log_test_cam = False
    eval_n_log_test_cam = True

    use_stree_grouping_strategy = True
    # use_stree_grouping_strategy = False

    if use_stree_grouping_strategy:
        use_streetgs_render = True #fail
        use_streetgs_render = False


    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*500 for i in range(0,120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[0,1])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "endonerf/pulling_fdm")
    parser.add_argument("--configs", type=str, default = "arguments/endonerf/default.py")

    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
        #update with tool_info automatically
        expname_append = ''
        if hasattr(args,'tool_mask'):
            expname_append += f'_{args.tool_mask}'
        if hasattr(args,'init_mode'):
            expname_append += f'_{args.init_mode}'
        if use_stree_grouping_strategy:
            assert args.tool_mask == 'use',f' for misgs,we let tool_mask be use n get all masks'

            #pose related setting
            expname_append += f'_{args.track_warmup_steps}_extent{args.camera_extent}_space{args.obj_pose_rot_optim_space}_{args.obj_pose_init}init'


        assert args.disable_tb in ['Y','N']
        if args.disable_tb == 'Y':
            expname_append += '_NOTB'

        setattr(args, 'expname', f'{args.expname}{expname_append}')
        
        if 'pulling' in args.source_path or 'cutting' in args.source_path:
            pass
            print('TODO','nouse and inverse might be problematic---the data gt depth always masked tool region....')
            # assert args.tool_mask == "use",'nouse and inverse might be problematic---the data gt depth always masked tool region....'
    else:
        assert 0

    args.save_iterations.append(args.iterations)
    args.model_path = args.expname#jj--use by misgs
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if not use_stree_grouping_strategy:
        training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, 
            extra_mark=args.extra_mark,
            args = args)
    else:
        # save args in model_path for offline rendering
        # Save the 'args' object to a file
        # from arguments import save_args
        # # already done by the author
        # # cfg_args file
        # save_args_path =os.path.join(args.model_path,'exp_default.py')
        # save_args(args, path = save_args_path)        
        
        #sanity
        assert args.obj_pose_init in ['0','cotrackerpnp']
        assert args.obj_pose_rot_optim_space in ['rpy','lie']
        if args.obj_pose_init in ['cotrackerpnp']:
            assert args.load_cotrackerPnpPose



        from train_utils_misgs import training_misgsmodel
        training_misgsmodel(args, use_streetgs_render = use_streetgs_render,
                                eval_n_log_test_cam = eval_n_log_test_cam,
                            )
    # All done
    print("\nTraining complete.", args.model_path)
