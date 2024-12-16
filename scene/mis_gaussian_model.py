from scene.flexible_deform_model import TissueGaussianModel 
from scene.tool_model import ToolModel
from scene.tool_pose import ToolPose
# from scene.poses_all_objs import ActorPose
import torch.nn as nn
import torch
import os
from bidict import bidict
from utils.general_utils import matrix_to_quaternion,\
    startswith_any,strip_symmetric,build_scaling_rotation,quaternion_to_matrix
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from scene.cameras import Camera
from scene.gaussian_model_base import GaussianModelBase
from plyfile import PlyData, PlyElement
from typing import Union
from utils.general_utils import quaternion_raw_multiply

class MisGaussianModel(nn.Module):
    def __init__(self, metadata,new_cfg):
        super().__init__()
        self.cfg = new_cfg
        self.metadata = metadata
            
        self.max_sh_degree =self.cfg.model.gaussian.sh_degree
        self.active_sh_degree = self.max_sh_degree

        self.include_tissue =self.cfg.model.nsg.include_tissue #get('include_tissue', True)
        self.include_obj =self.cfg.model.nsg.include_obj#get('include_obj', False) #False)
        self.include_obj_pose =self.cfg.model.nsg.include_obj_pose#get('include_obj', False) #False)
        self.include_background =self.cfg.model.nsg.include_bkgd#get('include_bkgd', False)
        self.include_sky =self.cfg.model.nsg.include_sky#get('include_sky', False) 

        if self.include_sky:
            assert self.cfg.data.white_background is False
        # fourier sh dimensions
        self.fourier_dim =self.cfg.model.gaussian.get('fourier_dim', 1)
        # layer color correction
        self.use_color_correction =self.cfg.model.use_color_correction
        # camera pose optimizations (not test)
        self.use_pose_correction =self.cfg.model.use_pose_correction
        # symmetry
        self.flip_prob =self.cfg.model.gaussian.get('flip_prob', 0.)
        self.flip_axis = 1 
        self.flip_matrix = torch.eye(3).float().cuda() * -1
        self.flip_matrix[self.flip_axis, self.flip_axis] = 1
        self.flip_matrix = matrix_to_quaternion(self.flip_matrix.unsqueeze(0))
        
        #//////////jj///////////////////
        # model names: manual add here!
        # save the same keys as self.model_name_id
        self.candidate_model_names = {}
        if self.include_background:
            self.candidate_model_names['bg_model'] = ['background']
            assert torch.Tensor([ name.startswith('background') for name in self.candidate_model_names['bg_model']]).all(),\
                f"not all names start_with background {self.candidate_model_names['bg_model']}"
            # assert len(self.candidate_model_names['bg_model'])==1,'later will use index[0]'
        if self.include_tissue:
            self.candidate_model_names['tissue_model'] = [
                                                        'tissue',
                                                        ]
            
            assert torch.Tensor([ name.startswith('tissue') for name in self.candidate_model_names['tissue_model']]).all(),\
                f"not all names start_with tissue {self.candidate_model_names['tissue_model']}"
        if self.include_obj:
            model_names_obj = [
                'obj_tool1'
            ]
            self.candidate_model_names['obj_model_cand']= model_names_obj
        #/////////////////////////////
        self.setup_functions() 
    
    def setup_functions(self):
        obj_tracklets = self.metadata['obj_tracklets']
        obj_info = self.metadata['obj_meta']
        tracklet_timestamps = self.metadata['tracklet_timestamps']
        camera_timestamps = self.metadata['camera_timestamps']
        
        self.model_name_id = bidict()
        self.obj_list = []
        self.obj_pose_list = []
        self.models_num = 0
        self.obj_info = obj_info
        # Build background model
        if self.include_background:
            model_names = self.candidate_model_names['bg_model']
            for model_name in model_names:
                model = GaussianModelBkgd(
                    model_name=model_name, 
                    scene_center=self.metadata['scene_center'],
                    scene_radius=self.metadata['scene_radius'],
                    sphere_center=self.metadata['sphere_center'],
                    sphere_radius=self.metadata['sphere_radius'],
                )
                setattr(self, model_name, model)
                self.model_name_id[model_name] = self.models_num
                self.models_num += 1

        # Build tissue model
        if self.include_tissue:
            model_names = self.candidate_model_names['tissue_model']
            for model_name in model_names:
                model = TissueGaussianModel(self.cfg.model.gaussian.sh_degree, \
                                                self.cfg.model.fdm)
                setattr(self, model_name, model )
                self.model_name_id[model_name] = self.models_num
                self.models_num += 1
        
        # Build object model
        self.poses_all_objs = None
        if self.include_obj:
            model_names = self.candidate_model_names['obj_model_cand']
            for i,model_name in enumerate(model_names):
                # ToolModel
                from scene.tool_model import ToolModel
                model = ToolModel(model_args = self.cfg.model.gaussian,
                                  obj_meta=None,
                                  track_id=i,
                                  cfg = self.cfg)
                setattr(self, model_name, model)
                self.model_name_id[model_name] = self.models_num
                self.models_num += 1
                self.obj_list.append(model_name)
            # Build actor model 
            from scene.tool_pose import ToolPose
            if self.include_obj_pose:
                # camera_timestamps contains train and val
                # frames_num is complet continous imgs
                self.poses_all_objs = ToolPose(
                                                objs_num=1, 
                                                camera_timestamps=camera_timestamps, 
                                                cfg_optim=self.cfg.optim,
                                                opt_track = self.cfg.model.nsg.opt_track,
                                                cam_id=0,
                                                cfg = self.cfg)
                self.obj_list.append(self.poses_all_objs)
                
    
    def set_visibility(self, include_list):
        self.include_list = include_list # prefix

    def get_visibility(self, model_name):
        if model_name.startswith('obj_'):
            if model_name in self.include_list and self.include_obj:
                return True
            else:
                return False
        elif model_name.startswith('tissue'):
        # elif model_name == 'tissue':
            if model_name in self.include_list and self.include_tissue:
                return True
            else:
                return False
        else:
            raise ValueError(f'Unknown model name {model_name}')
    def create_from_pcd(self, 
                        pcd_dict: dict, 
                        spatial_lr_scale: float,\
                        time_line: int):# FDM need
        for model_name in self.model_name_id.keys():
            if model_name.startswith('tissue'):
                model: TissueGaussianModel = getattr(self, model_name)
                print('Try to be the same at first')
                model.create_from_pcd(pcd = pcd_dict[model_name], 
                                      spatial_lr_scale = spatial_lr_scale, 
                                      time_line = time_line)
            elif model_name.startswith('obj_'):
                model: ToolModel = getattr(self, model_name)
                model.create_from_pcd(pcd = pcd_dict[model_name], 
                                      spatial_lr_scale = spatial_lr_scale, 
                                      time_line = time_line)
            else:
                assert 0, model_name
    def save_ply(self, path, 
                 ):
        mkdir_p(os.path.dirname(path))
        plydata_list = []
        for i in range(self.models_num):
            model_name = self.model_name_id.inverse[i]
            model: GaussianModelBase = getattr(self, model_name)
            if os.path.exists(path):
                assert 0,f'{model_name} {model} {path}'
            try:
                plydata = model.make_ply()
                plydata = PlyElement.describe(plydata, f'vertex_{model_name}')
            except:
                assert isinstance(model,TissueGaussianModel) or isinstance(model,ToolModel)
                plydata = model.save_ply(path = path,
                                         only_make = True)
                plydata = PlyElement.describe(plydata, f'vertex_{model_name}')
                
                # save ply model_wise for debug
                model.save_ply(path = path.replace('.ply',f'_{model_name}.ply'), only_make = False)



            plydata_list.append(plydata)
        PlyData(plydata_list).write(path)
        
    def load_ply(self, path):
        plydata_list = PlyData.read(path).elements
        for plydata in plydata_list:
            model_name = plydata.name[7:] # vertex_.....
            if model_name in self.model_name_id.keys():
                print('Loading model', model_name)
                model: Union[GaussianModelBase,TissueGaussianModel] = getattr(self, model_name)
                model.load_ply(path=None, input_ply=plydata)
                plydata_list = PlyData.read(path).elements
        self.active_sh_degree = self.max_sh_degree
  
    def load_state_dict(self, state_dict, exclude_list=[]):
        assert 0,'not tested'
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModelBase = getattr(self, model_name)
            model.load_state_dict(state_dict[model_name])
        
        if self.poses_all_objs is not None:
            self.poses_all_objs.load_state_dict(state_dict['poses_all_objs'])
        # if self.sky_cubemap is not None:
        #     self.sky_cubemap.load_state_dict(state_dict['sky_cubemap'])
        # if self.color_correction is not None:
        #     self.color_correction.load_state_dict(state_dict['color_correction'])
        # if self.pose_correction is not None:
        #     self.pose_correction.load_state_dict(state_dict['pose_correction'])
                            
    def save_state_dict(self, is_final, exclude_list=[]):
        state_dict = dict()
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: Union[GaussianModelBase,TissueGaussianModel] = getattr(self, model_name)
            state_dict[model_name] = model.state_dict(is_final)
        
        # if self.poses_all_objs is not None:
            # state_dict['poses_all_objs'] = self.poses_all_objs.save_state_dict(is_final)


        # if self.sky_cubemap is not None:
        #     state_dict['sky_cubemap'] = self.sky_cubemap.save_state_dict(is_final)
        # if self.color_correction is not None:
        #     state_dict['color_correction'] = self.color_correction.save_state_dict(is_final)
        # if self.pose_correction is not None:
        #     state_dict['pose_correction'] = self.pose_correction.save_state_dict(is_final)
      
        return state_dict
        
    def parse_camera(self, camera: Camera):
        ''''
        maintain: 
        self.num_gaussians 
        self.graph_gaussian_range
        self.graph_obj_list
        more ?
        '''
        # set camera
        self.viewpoint_camera = camera
        self.frame = camera.meta['frame']
        self.frame_idx = camera.meta['frame_idx']
        self.frame_is_val = camera.meta['is_val']
        self.num_gaussians = 0

        #///////////////////////jj
        # object (build scene graph)
        # an obj model can have general True visinility but due to timestamp issue not included in graph_obj_list
        self.graph_obj_list = [] 
        # set index range
        # self.graph_gaussian_range = dict()
        idx = 0
        for model_name in self.model_name_id.keys():
            if self.get_visibility(model_name=model_name):
                model = getattr(self, model_name)
                if model_name.startswith('obj_'):
                    assert self.include_obj
                    timestamp = camera.meta['timestamp']
                    # assert 0,timestamp
                    # start_timestamp, end_timestamp = model.start_timestamp, model.end_timestamp
                    # if timestamp >= start_timestamp and timestamp <= end_timestamp and self.get_visibility(obj_name):
                    self.num_gaussians += model.get_xyz.shape[0]
                    self.graph_obj_list.append(model_name)
                else:
                    self.num_gaussians += model.get_xyz.shape[0]

                # self.graph_gaussian_range[model_name] = [idx, idx+model.get_xyz.shape[0]-1]
                idx += model.get_xyz.shape[0]


        #/////////////////////
        # for model_name in self.model_name_id.keys():
        #     if self.get_visibility(model_name=model_name):
        #         if model_name.startswith('obj_') and model_name not in self.graph_obj_list:
        #             continue
        #         num_gaussians = getattr(self, model_name).get_xyz.shape[0]
        #         self.num_gaussians += num_gaussians
        #////////////////////////
        # if len(self.graph_obj_list) > 0:
        if len(self.graph_obj_list) > 0 and self.include_obj_pose:
            self.obj_rots = []
            self.obj_trans = []
            for i, obj_name in enumerate(self.graph_obj_list):
                obj_model: ToolModel = getattr(self, obj_name)
                track_id = obj_model.track_id
                assert track_id==0
                obj_rot = self.poses_all_objs.get_tracking_rotation(track_id, self.viewpoint_camera)
                # it will use the trans info of the next two frames
                obj_trans = self.poses_all_objs.get_tracking_translation(track_id, self.viewpoint_camera)  
                # internally call:  get_tracking_translation_(self, track_id, timestamp)
                # which learn the drift only--- 
                ego_pose = self.viewpoint_camera.ego_pose
                ego_pose_rot = matrix_to_quaternion(ego_pose[:3, :3].unsqueeze(0)).squeeze(0)
                obj_rot = quaternion_raw_multiply(ego_pose_rot.unsqueeze(0), obj_rot.unsqueeze(0)).squeeze(0)
                obj_trans = ego_pose[:3, :3] @ obj_trans + ego_pose[:3, 3]
                
                obj_rot = obj_rot.expand(obj_model.get_xyz.shape[0], -1)
                obj_trans = obj_trans.unsqueeze(0).expand(obj_model.get_xyz.shape[0], -1)
                
                self.obj_rots.append(obj_rot)
                self.obj_trans.append(obj_trans)
            
            self.obj_rots = torch.cat(self.obj_rots, dim=0)
            self.obj_trans = torch.cat(self.obj_trans, dim=0)  
            
            self.flip_mask = []
            for obj_name in self.graph_obj_list:
                obj_model: ToolModel = getattr(self, obj_name)
                flip_mask = torch.zeros_like(obj_model.get_xyz[:, 0]).bool()
                self.flip_mask.append(flip_mask)
            self.flip_mask = torch.cat(self.flip_mask, dim=0)   
            
    @property
    def get_scaling(self):
        scalings = []
        for model_name in self.model_name_id.keys():
            if self.get_visibility(model_name=model_name):
                scaling = getattr(self, model_name).get_scaling
                scalings.append(scaling)
        scalings = torch.cat(scalings, dim=0)
        return scalings
            

    @property
    def get_rotation_obj_only(self):
        rotations = []
        # process obj pose
        rotations_local = []
        for i, obj_name in enumerate(self.graph_obj_list):
            assert self.get_visibility(model_name=obj_name)
            rotations_local.append(getattr(self, obj_name).get_rotation)
        if len(self.graph_obj_list) > 0:
            rotations_local = torch.cat(rotations_local, dim=0)
            rotations_local = rotations_local.clone()
            rotations_local[self.flip_mask] = quaternion_raw_multiply(self.flip_matrix, rotations_local[self.flip_mask])
            rotations_obj = quaternion_raw_multiply(self.obj_rots, rotations_local)
            rotations_obj = torch.nn.functional.normalize(rotations_obj)
            rotations.append(rotations_obj)

        rotations = torch.cat(rotations, dim=0)
        return rotations
    
    @property
    def get_xyz_obj_only(self):
        xyzs = []
        # # # process obj pose
        xyzs_local = []
        for i, obj_name in enumerate(self.graph_obj_list):
            assert self.get_visibility(model_name=obj_name)
            xyzs_local.append(getattr(self, obj_name).get_xyz)
        if len(self.graph_obj_list) > 0:
            xyzs_local = torch.cat(xyzs_local, dim=0)
            xyzs_local = xyzs_local.clone()
            xyzs_local[self.flip_mask, self.flip_axis] *= -1
            obj_rots = quaternion_to_matrix(self.obj_rots)
            xyzs_obj = torch.einsum('bij, bj -> bi', obj_rots, xyzs_local) + self.obj_trans
            xyzs.append(xyzs_obj) 
    
        xyzs = torch.cat(xyzs, dim=0)
        return xyzs            

    


    @property
    def get_rotation(self):
        assert 0,'only implement below when use misgs render'
        rotations = []
        for model_name in self.model_name_id.keys():
            if self.get_visibility(model_name=model_name):
                rotation = getattr(self, model_name).get_rotation
                rotations.append(rotation)

        # process obj pose
        rotations_local = []
        for i, obj_name in enumerate(self.graph_obj_list):
            assert self.get_visibility(model_name=obj_name)
            rotations_local.append(getattr(self, obj_name).get_rotation)
        if len(self.graph_obj_list) > 0:
            rotations_local = torch.cat(rotations_local, dim=0)
            rotations_local = rotations_local.clone()
            rotations_local[self.flip_mask] = quaternion_raw_multiply(self.flip_matrix, rotations_local[self.flip_mask])
            rotations_obj = quaternion_raw_multiply(self.obj_rots, rotations_local)
            rotations_obj = torch.nn.functional.normalize(rotations_obj)
            rotations.append(rotations_obj)

        rotations = torch.cat(rotations, dim=0)
        return rotations
    
    @property
    def get_xyz(self):
        assert 0,'only implement below when use misgs render'
        # first tissue then obj(tool)
        xyzs = []
        for model_name in self.model_name_id.keys():
            if self.get_visibility(model_name=model_name):
                if isinstance(getattr(self, model_name),TissueGaussianModel):
                    xyz = getattr(self, model_name).get_xyz
                xyzs.append(xyz)

        # # # process obj pose
        xyzs_local = []
        for i, obj_name in enumerate(self.graph_obj_list):
            assert self.get_visibility(model_name=obj_name)
            xyzs_local.append(getattr(self, obj_name).get_xyz)
        if len(self.graph_obj_list) > 0:
            xyzs_local = torch.cat(xyzs_local, dim=0)
            xyzs_local = xyzs_local.clone()
            xyzs_local[self.flip_mask, self.flip_axis] *= -1
            obj_rots = quaternion_to_matrix(self.obj_rots)
            xyzs_obj = torch.einsum('bij, bj -> bi', obj_rots, xyzs_local) + self.obj_trans
            xyzs.append(xyzs_obj) 
    
        xyzs = torch.cat(xyzs, dim=0)
        return xyzs            
    @property
    def get_features(self):
        features = []
        for model_name in self.model_name_id.keys():
            if self.get_visibility(model_name=model_name):
                feature = getattr(self, model_name).get_features
                features.append(feature)
        features = torch.cat(features, dim=0)
        return features
    
    def get_colors(self, camera_center):
        colors = []
        for model_name in self.model_name_id.keys():
            if self.get_visibility(model_name=model_name):
                model= getattr(self, model_name)
                max_sh_degree = model.max_sh_degree
                sh_dim = (max_sh_degree + 1) ** 2

                if model_name.startswith('tissue'):                  
                    shs = model.get_features.transpose(1, 2).view(-1, 3, sh_dim)
                elif model_name.startswith('obj_tool'):                  
                    features = model.get_features_fourier(self.frame)
                    shs = features.transpose(1, 2).view(-1, 3, sh_dim)
                else:
                    assert 0,model_name

                directions = model.get_xyz - camera_center
                directions = directions / torch.norm(directions, dim=1, keepdim=True)
                from utils.sh_utils import eval_sh
                sh2rgb = eval_sh(max_sh_degree, shs, directions)
                color = torch.clamp_min(sh2rgb + 0.5, 0.)
                colors.append(color)
        colors = torch.cat(colors, dim=0)
        return colors
                

    @property
    def get_opacity(self):
        opacities = []
        for model_name in self.model_name_id.keys():
            if self.get_visibility(model_name=model_name):
                opacity = getattr(self, model_name).get_opacity
                opacities.append(opacity)
        opacities = torch.cat(opacities, dim=0)
        return opacities
            
        
    def oneupSHdegree(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if model_name in exclude_list:
                continue
            model: Union[GaussianModelBase,TissueGaussianModel] = getattr(self, model_name)
            model.oneupSHdegree()
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, exclude_list=[]):
        self.active_sh_degree = 0
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: Union[GaussianModelBase,TissueGaussianModel,ToolModel] = getattr(self, model_name)
            if model_name.startswith('tissue'):
                model.training_setup(training_args=self.cfg.optim)
            elif model_name.startswith('obj_'):
                model.training_setup(training_args=self.cfg.optim)
            else:
                assert 0,NotImplementedError
                model.training_setup()
                
        if self.poses_all_objs is not None:
            self.poses_all_objs.training_setup()
        # if self.sky_cubemap is not None:
        #     assert 0
        #     self.sky_cubemap.training_setup()
        # if self.color_correction is not None:
        #     assert 0
        #     self.color_correction.training_setup()
        # if self.pose_correction is not None:
        #     assert 0
        #     self.pose_correction.training_setup()
        
    def update_learning_rate(self, iteration, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            # model: GaussianModelBase = getattr(self, model_name)
            model: Union[GaussianModelBase,TissueGaussianModel] = getattr(self, model_name)
            model.update_learning_rate(iteration)
        
        if self.poses_all_objs is not None:
            self.poses_all_objs.update_learning_rate(iteration)
        # if self.sky_cubemap is not None:
        #     self.sky_cubemap.update_learning_rate(iteration)
        # if self.color_correction is not None:
        #     self.color_correction.update_learning_rate(iteration)
        # if self.pose_correction is not None:
        #     self.pose_correction.update_learning_rate(iteration)
    
    def update_optimizer(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: Union[GaussianModelBase,TissueGaussianModel] = getattr(self, model_name)
            model.update_optimizer()

        if self.poses_all_objs is not None:
            self.poses_all_objs.update_optimizer()
        # if self.sky_cubemap is not None:
        #     self.sky_cubemap.update_optimizer()
        # if self.color_correction is not None:
        #     self.color_correction.update_optimizer()
        # if self.pose_correction is not None:
        #     self.pose_correction.update_optimizer()

    # def set_max_radii2D(self, radii, visibility_filter):
    #     '''
    #     already internnallly performed by the densify and prune of tissue model
    #     '''
    #     radii = radii.float()
        
    #     for model_name in self.graph_gaussian_range.keys():
    #         model: GaussianModelBase = getattr(self, model_name)
    #         start, end = self.graph_gaussian_range[model_name]
    #         end += 1
    #         visibility_model = visibility_filter[start:end]
    #         max_radii2D_model = radii[start:end]



    #         model.max_radii2D[visibility_model] = torch.max(
    #             model.max_radii2D[visibility_model], max_radii2D_model[visibility_model])
        
    
    
    def set_max_radii2D_all_models(self, radiis = [], visibility_filters = [],model_names = []):
        '''
        already internnallly performed by the densify and prune of tissue model
        '''

        assert len(radiis) == len(visibility_filters)
        assert len(radiis) == len(model_names)

        for radii, visibility_filter, model_name in zip(radiis,visibility_filters,model_names):
            radii = radii.float()
            model: GaussianModelBase = getattr(self, model_name)
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter], radii[visibility_filter])
        
    
    
    
    # def add_densification_stats(self, viewspace_point_tensor, visibility_filter):
    #     '''
    #     already internnallly performed by the densify and prune of tissue model
    #     '''

    #     # assert 0,'not checked'
    #     viewspace_point_tensor_grad = viewspace_point_tensor.grad
    #     for model_name in self.graph_gaussian_range.keys():
    #         model: GaussianModelBase = getattr(self, model_name)
    #         start, end = self.graph_gaussian_range[model_name]
    #         end += 1
    #         visibility_model = visibility_filter[start:end]
    #         viewspace_point_tensor_grad_model = viewspace_point_tensor_grad[start:end]
    #         model.xyz_gradient_accum[visibility_model, 0:1] += torch.norm(viewspace_point_tensor_grad_model[visibility_model, :2], dim=-1, keepdim=True)
    #         model.xyz_gradient_accum[visibility_model, 1:2] += torch.norm(viewspace_point_tensor_grad_model[visibility_model, 2:], dim=-1, keepdim=True)
    #         model.denom[visibility_model] += 1

    def add_densification_stats_all_models(self, viewspace_point_tensors, visibility_filters, model_names):
        '''
        already internnallly performed by the densify and prune of tissue model
        '''

        assert len(viewspace_point_tensors) == len(model_names)
        assert len(visibility_filters) == len(model_names)


        for viewspace_point_tensor, visibility_filter, model_name in zip(viewspace_point_tensors,visibility_filters,model_names):
            # assert 0,'not checked'
            viewspace_point_tensor_grad = viewspace_point_tensor.grad
            model: GaussianModelBase = getattr(self, model_name)
            model.xyz_gradient_accum[visibility_filter, 0:1] += torch.norm(viewspace_point_tensor_grad[visibility_filter, :2], dim=-1, keepdim=True)
            model.xyz_gradient_accum[visibility_filter, 1:2] += torch.norm(viewspace_point_tensor_grad[visibility_filter, 2:], dim=-1, keepdim=True)
            model.denom[visibility_filter] += 1

    
    
    def reset_opacity(self, exclude_list=[]):
        # assert 0
        '''
        already internnallly performed by the densify and prune of tissue model
        '''
        for model_name in self.model_name_id.keys():
            model: GaussianModelBase = getattr(self, model_name)
            if startswith_any(model_name, exclude_list):
                continue
            model.reset_opacity()


    # def densify_and_prune(self, max_grad, min_opacity, prune_big_points, exclude_list=[],extent = None,max_screen_size = None):
    def densify_and_prune(self, 
                          max_grad = None, 
                          min_opacity = None,
                          exclude_list=[],
                          extent = None,
                          max_screen_size = None,
                          skip_densify = None,
                          skip_prune = None,
                          percent_big_ws = None,
                          ):
        scalars = {}#None
        tensors = {}#None
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            # if model_name == 'tissue':
            model: Union[TissueGaussianModel,ToolModel] = getattr(self, model_name)
            if isinstance(model,TissueGaussianModel) \
                    :

                # or isinstance(model,ToolModel)\

                scalars_, tensors_ = model.densify_and_prune(max_grad = max_grad, 
                                                            min_opacity = min_opacity, 
                                                            extent=extent, 
                                                            max_screen_size=max_screen_size,
                                                            skip_densify=skip_densify,
                                                            skip_prune=skip_prune,
                                                            #  percent_big_ws=percent_big_ws,
                                                            )
            elif isinstance(model,ToolModel):
                # assert 0, 'debug'
                scalars_, tensors_ = model.densify_and_prune(max_grad = max_grad, 
                                                            min_opacity = min_opacity, 
                                                            extent=extent, 
                                                            K=torch.Tensor(self.viewpoint_camera.K),
                                                            init_tool_mask=torch.Tensor(self.metadata['init_mask_dict']['obj_tool1']).to(torch.bool),#.unsqueeze(0),
                                                            # init_tool_mask=torch.Tensor(self.viewpoint_camera.tool_mask),
                                                            current_tool_mask = torch.Tensor(self.viewpoint_camera.tool_mask).squeeze(0),
                                                            )

            
 



            if model_name == 'background':
                scalars = scalars_
                tensors = tensors_
    
        return scalars, tensors
    

    def get_box_reg_loss(self):
        box_reg_loss = 0.
        for obj_name in self.obj_list:
            assert 0
            obj_model: ToolPose = getattr(self, obj_name)
            box_reg_loss += obj_model.box_reg_loss()
        box_reg_loss /= len(self.obj_list)

        return box_reg_loss
            
