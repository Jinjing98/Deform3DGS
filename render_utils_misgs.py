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
import imageio
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_flow as render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, FDMHiddenParams
from scene.flexible_deform_model import TissueGaussianModel
from scene.tool_model import ToolModel
from time import time
import open3d as o3d
from utils.graphics_utils import fov2focal
import cv2
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set_misgs(model_path, name, iteration, views, gaussians, pipeline, background,\
    no_fine, render_test=False, reconstruct=False, crop_size=0,
    which_model = 'tissue',
    ):
    from render import reconstruct_point_cloud
    if which_model == 'tissue':
        render_func = render
    elif which_model == 'obj_tool1':
        from gaussian_renderer.tool_renderer import tool_render
        render_func = tool_render
        assert 0,'to do'
    elif which_model == 'all':
        from gaussian_renderer.misgs_renderer import MisGaussianRenderer
        render_func = MisGaussianRenderer
    else:
        assert 0, NotImplementedError
    # #/////////////////////////
    # try:
    #     assert 'tissue' in model_name
    #     render_pkg= fdm_render(viewpoint_cam, sub_gs_model, cfg.render, background)
    # except:
            # tool render need misgs model due to tool_pose learning
    #     assert 'tool' in model_name
    #     render_pkg= tool_render(viewpoint_cam, sub_gs_model, cfg.render, background,
    #                             debug_getxyz_misgs = debug_getxyz_misgs,
    #                             misgs_model = controller,
    #                             )
    # #/////////////////////////

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"renders_{which_model}")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"depth_{which_model}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_{which_model}")
    gtdepth_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_depth_{which_model}")
    masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"masks_{which_model}")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gtdepth_path, exist_ok=True)
    makedirs(masks_path, exist_ok=True)
    
    render_images = []
    render_depths = []
    gt_list = []
    gt_depths = []
    mask_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        stage = 'coarse' if no_fine else 'fine'
        rendering = render_func(view, gaussians, pipeline, background)
        render_depths.append(rendering["depth"].cpu())
        render_images.append(rendering["render"].cpu())
        if name in ["train", "test", "video"]:
            gt = view.original_image[0:3, :, :]
            gt_list.append(gt)
            mask = view.mask
            mask_list.append(mask)
            gt_depth = view.original_depth
            gt_depths.append(gt_depth)
    
    if render_test:
        test_times = 20
        for i in range(test_times):
            for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
                if idx == 0 and i == 0:
                    time1 = time()
                stage = 'coarse' if no_fine else 'fine'
                rendering = render_func(view, gaussians, pipeline, background)
        time2=time()
        print("FPS:",(len(views)-1)*test_times/(time2-time1))
    
    count = 0
    print("writing training images.")
    if len(gt_list) != 0:
        for image in tqdm(gt_list):
            torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
            count+=1
            
    count = 0
    print("writing rendering images.")
    if len(render_images) != 0:
        for image in tqdm(render_images):
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing mask images.")
    if len(mask_list) != 0:
        for image in tqdm(mask_list):
            image = image.float()
            torchvision.utils.save_image(image, os.path.join(masks_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing rendered depth images.")
    if len(render_depths) != 0:
        for image in tqdm(render_depths):
            image = np.clip(image.cpu().squeeze().numpy().astype(np.uint8), 0, 255)
            cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(count) + ".png"), image)
            count += 1
    
    count = 0
    print("writing gt depth images.")
    if len(gt_depths) != 0:
        for image in tqdm(gt_depths):
            image = image.cpu().squeeze().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(gtdepth_path, '{0:05d}'.format(count) + ".png"), image)
            count += 1
            
    render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255).cpu().numpy().astype(np.uint8) # BxHxWxC
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'ours_video.mp4'), render_array, fps=30, quality=8)
    
    gt_array = torch.stack(gt_list, dim=0).permute(0, 2, 3, 1)
    gt_array = (gt_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'gt_video.mp4'), gt_array, fps=30, quality=8)
                    
    FoVy, FoVx, height, width = view.FoVy, view.FoVx, view.image_height, view.image_width
    focal_y, focal_x = fov2focal(FoVy, height), fov2focal(FoVx, width)
    camera_parameters = (focal_x, focal_y, width, height)
    

    if reconstruct:
        print('file name:', name)
        reconstruct_point_cloud(render_images, mask_list, render_depths, camera_parameters, name, crop_size)

def render_sets_misgs(
        mod_stree_param : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, 
                skip_train : bool, skip_test : bool, skip_video: bool, 
                reconstruct_train: bool, reconstruct_test: bool, reconstruct_video: bool,
                which_model = 'tissue',
                cfg_path = '', 
                ):
    from scene.mis_gaussian_model import MisGaussianModel
    from config.yacs import load_cfg
    with open(cfg_path, 'r') as f:
        cfg = load_cfg(f)

    if which_model in ['tissue','obj_tool1']:
        load_which_pcd = 'point_cloud_tissue' if which_model in ['tissue'] else 'point_cloud_obj_tool1'
        gaussians = TissueGaussianModel(mod_stree_param.sh_degree, hyperparam)  if which_model in ['tissue'] else \
            ToolModel(mod_stree_param.sh_degree, hyperparam)
        scene = Scene(mod_stree_param)
        assert 0, f'need load_other_obj_meta for pose?'
        scene = Scene(mod_stree_param,load_other_obj_meta=True,new_cfg=cfg)

        scene.gs_init(gaussians_or_controller=gaussians, load_iteration=iteration,
                        reset_camera_extent=mod_stree_param.camera_extent,
                        load_which_pcd=load_which_pcd)
        # render_set_func = render_set if which_model in ['tissue'] else render_set_misgs
    elif which_model == 'all':
        load_which_pcd = 'point_cloud'
        scene = Scene(mod_stree_param,load_other_obj_meta=True,new_cfg=cfg)
        controller = MisGaussianModel(metadata=scene.getSceneMetaData(),
                                    new_cfg=cfg)#nn.module instance
        print('todo singel render need to be implemented')
        # render_set_func = render_set_misgs
        assert 0, NotImplementedError
    else:
        assert 0, NotImplementedError

    from scene.mis_gaussian_model import MisGaussianModel
    assert which_model in ['tissue','obj_tool1','all']
    load_other_obj_meta=True #load within the sceneinfo
    scene = Scene(mod_stree_param,load_other_obj_meta=True,new_cfg=cfg)
    
    controller = MisGaussianModel(metadata=scene.getSceneMetaData(),
                                 new_cfg=cfg)#nn.module instance
    
    bg_color = [1,1,1] if mod_stree_param.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    
    if not skip_train:
        render_set_misgs(mod_stree_param.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, False, reconstruct=reconstruct_train,
                         which_model=which_model)
    if not skip_test:
        render_set_misgs(mod_stree_param.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, False, reconstruct=reconstruct_test, crop_size=20,
                         which_model=which_model)
    if not skip_video:
        render_set_misgs(mod_stree_param.model_path,"video",scene.loaded_iter, scene.getVideoCameras(),gaussians,pipeline,background, False, render_test=True, reconstruct=reconstruct_video, crop_size=20,
                         which_model=which_model)


 
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = FDMHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--reconstruct_train", action="store_true")
    parser.add_argument("--reconstruct_test", action="store_true")
    parser.add_argument("--reconstruct_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser) # this file will merge the saved args for this exp; the same func as exp_default.py
    print("Rendering ", args.model_path)
    render_misgs = True
    # exp_time_args_file_name = 'exp_default.py'
    # if exp_time_args_file_name not in args.configs:
    #     print('Wrong config, update with exp-time snapshot...')
    #     args.configs = os.path.join(args.model_path,exp_time_args_file_name)
    #     assert os.path.exists(args.configs),f'not saved args during traing? {args.configs}'

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    if not render_misgs:
        assert 0
    else:
        from render import render_sets_misgs
        exp_time_cfg_file_name = 'configs/config_000000.yaml'
        cfg_path = os.path.join(args.model_path,exp_time_args_file_name)
        assert os.path.exists(cfg_path),f'not saved cfg during traing? {args.configs}'
        render_sets_misgs(model.extract(args), hyperparam.extract(args), args.iteration, 
            pipeline.extract(args), 
            args.skip_train, args.skip_test, args.skip_video,
            args.reconstruct_train,args.reconstruct_test,args.reconstruct_video,
            which_model='tissue',
            cfg_path=cfg_path)