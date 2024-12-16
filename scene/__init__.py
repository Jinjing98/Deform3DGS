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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.flexible_deform_model import TissueGaussianModel
from scene.tool_model import ToolModel
from arguments import ModelParams

from typing import Union
# from gaussian_model_base import GaussianModelBase
from scene.mis_gaussian_model import MisGaussianModel
from config.argsgroup_misgs import ModParams
import torch.nn as nn


class Scene:

    # gaussians : TissueGaussianModel
    # gaussians : Union[GaussianModelBase, MisGaussianModel]
    gaussians_or_controller : Union[TissueGaussianModel, MisGaussianModel]
    
    def __init__(self, \
                #  args : ModelParams,
                 args : Union[ModelParams,ModParams],#Dataparams used for load scene_meta
                 load_other_obj_meta = False,
                 new_cfg = None,
                 load_pcd_dict_in_sceneinfo = False,
                 ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.load_pcd_dict_in_sceneinfo = load_pcd_dict_in_sceneinfo

        self.model_path = args.model_path
        if os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")) and args.extra_mark == 'endonerf':
            scene_info = sceneLoadTypeCallbacks["endonerf"](args.source_path,
                                                            tool_mask=args.tool_mask,
                                                            init_mode=args.init_mode,
                                                            load_other_obj_meta=load_other_obj_meta,
                                                            cfg = new_cfg,
                                                            load_pcd_dict_in_sceneinfo=self.load_pcd_dict_in_sceneinfo,
                                                            
                                                            )
            print("Found poses_bounds.py and extra marks with EndoNeRf")
        elif os.path.exists(os.path.join(args.source_path, "point_cloud.obj")) or os.path.exists(os.path.join(args.source_path, "left_point_cloud.obj")):
            assert 0,'not for misgs yet'
            assert self.tool_mask == 'use', NotImplementedError
            scene_info = sceneLoadTypeCallbacks["scared"](args.source_path, args.white_background, args.eval)
            print("Found point_cloud.obj, assuming SCARED data!")
        else:
            assert 0,'not for misgs yet'
            assert False, "Could not recognize scene type!"
                
        self.maxtime = scene_info.maxtime

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.cameras_extent = 10
        self.cameras_extent = 0
        # assert 0,self.cameras_extent
        self.cameras_extent = args.camera_extent
        # print("self.cameras_extent is ", self.cameras_extent)



        print("Loading Training Cameras")
        self.train_camera = scene_info.train_cameras 
        print("Loading Test Cameras")
        self.test_camera = scene_info.test_cameras 
        print("Loading Video Cameras")
        self.video_camera =  scene_info.video_cameras 
        
        # assert 0, f'{scene_info.point_cloud.points} {len(scene_info.point_cloud.points)}'
        # xyz_max = scene_info.point_cloud.points.max(axis=0)
        # xyz_min = scene_info.point_cloud.points.min(axis=0)
        # self.gaussians_or_controller._deformation.deformation_net.grid.set_aabb(xyz_max,xyz_min)
        #jj
        print("Loading point_cloud (use for can step-step train init)")
        self.point_cloud =  scene_info.point_cloud 
        self.point_cloud_dict =  scene_info.point_cloud_dict 

        self.scene_metadata = scene_info.scene_metadata

        self.loaded_iter = None
        self.gaussians_or_controller = None



    def gs_init(self,gaussians_or_controller : Union[TissueGaussianModel, MisGaussianModel],\
                load_iteration=None,
                reset_camera_extent = None,
                ):
        # self.loaded_iter = None
        self.gaussians_or_controller = gaussians_or_controller
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        if self.loaded_iter:
            self.gaussians_or_controller.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians_or_controller.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                   ))
        else:
            if self.load_pcd_dict_in_sceneinfo:
                # depending on the spercific init_strategy: it will update self.point_cloud
                assert self.point_cloud_dict!={}
                assert isinstance(self.gaussians_or_controller,nn.Module)
                self.gaussians_or_controller.create_from_pcd(pcd_dict=self.point_cloud_dict, 
                                                             spatial_lr_scale=reset_camera_extent, 
                                                             time_line=self.maxtime)
            else:
                assert self.point_cloud!=None
                assert isinstance(self.gaussians_or_controller, TissueGaussianModel) \
                    or isinstance(self.gaussians_or_controller, ToolModel) 
                 #gs model
                self.gaussians_or_controller.create_from_pcd(pcd=self.point_cloud, 
                                                             spatial_lr_scale=reset_camera_extent, 
                                                             time_line=self.maxtime)

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians_or_controller.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # self.gaussians_or_controller.save_deformation(point_cloud_path)
    
    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        return self.video_camera
    
    def getSceneMetaData(self):
        return self.scene_metadata
    
