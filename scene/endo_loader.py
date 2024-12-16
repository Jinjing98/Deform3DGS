import warnings

warnings.filterwarnings("ignore")

import json
import os
import random
import os.path as osp

import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from utils.graphics_utils import focal2fov, fov2focal
import glob
from torchvision import transforms as T
# import open3d as o3d
from tqdm import trange
import imageio.v2 as iio
import cv2
import copy
import torch
import torch.nn.functional as F
from utils.general_utils import inpaint_depth, inpaint_rgb
from utils.sh_utils import SH2RGB

def generate_se3_matrix(translation, rotation_rad):


    # Create rotation matrices around x, y, and z axes
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotation_rad[0]), -np.sin(rotation_rad[0])],
                   [0, np.sin(rotation_rad[0]), np.cos(rotation_rad[0])]])

    Ry = np.array([[np.cos(rotation_rad[1]), 0, np.sin(rotation_rad[1])],
                   [0, 1, 0],
                   [-np.sin(rotation_rad[1]), 0, np.cos(rotation_rad[1])]])

    Rz = np.array([[np.cos(rotation_rad[2]), -np.sin(rotation_rad[2]), 0],
                   [np.sin(rotation_rad[2]), np.cos(rotation_rad[2]), 0],
                   [0, 0, 1]])

    # Combine rotations
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Create S(3) matrix
    se3_matrix = np.eye(4)


    se3_matrix[:3, :3] = R
    se3_matrix[:3, 3] = translation

    return se3_matrix

def update_extr(c2w, rotation_deg, radii_mm):
        rotation_rad = np.radians(rotation_deg)
        translation = np.array([-radii_mm * np.sin(rotation_rad) , 0, radii_mm * (1 - np.cos(rotation_rad))])
        # translation = np.array([0, 0, 10])
        se3_matrix = generate_se3_matrix(translation, (0,rotation_rad,0)) # transform_C_C'
        extr = np.linalg.inv(se3_matrix) @ np.linalg.inv(c2w) # transform_C'_W = transform_C'_C @ (transform_W_C)^-1
        
        return np.linalg.inv(extr) # c2w

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)







class EndoNeRF_Dataset(object):
    def __init__(
        self,
        datadir,
        downsample=1.0,
        test_every=8,
        tool_mask = 'use',

    ):
        self.img_wh = (
            int(640 / downsample),
            int(512 / downsample),
        )
        self.root_dir = datadir
        self.downsample = downsample 
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()
        self.white_bg = False
        #extend
        self.tool_mask = tool_mask
        if 'pulling' in self.root_dir or 'cutting' in self.root_dir:
            self.dataset = 'EndoNeRF' 
        elif 'P2_' in self.root_dir:
            self.dataset = "StereoMIS"
        else:
            assert 0, self.root_dir

        self.load_meta()
        print(f"Scene_Meta_data+Cam_K_T loaded, total image:{len(self.image_paths)}")
        
        n_frames = len(self.image_paths)
        self.train_idxs = [i for i in range(n_frames) if (i-1) % test_every != 0]
        self.test_idxs = [i for i in range(n_frames) if (i-1) % test_every == 0]
        self.all_idxs = [i for i in range(n_frames)]
        self.video_idxs = self.all_idxs#self.test_idxs #[i for i in range(n_frames)]
        #jj

        self.camera_timestamps = {"0":{"train_timestamps": self.train_idxs,\
                                "test_timestamps": self.test_idxs,
                                "video_timestamps": self.video_idxs,
                                "all_timestamps": self.all_idxs,
                                }}

        self.maxtime = 1.0

    def load_meta(self):
        """
        Load Scene_Meta_data+Cam_K_T loadedfrom the dataset.
        """
        
        # coordinate transformation 
        if self.dataset == 'StereoMIS':
            poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
            try:
                poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
            except: 
                # No far and near
                poses = poses_arr.reshape([-1, 3, 5])  # (N_cams, 3, 5)
            # StereoMIS has well calibrated intrinsics
            cy, cx, focal =  poses[0, :, -1]
            cy = 512//2
            cx = 640//2
            focal = focal / self.downsample
            self.focal = (focal, focal)
            self.K = np.array([[focal, 0 , cx],
                                        [0, focal, cy],
                                        [0, 0, 1]]).astype(np.float32)
            poses = np.concatenate([poses[..., :1], poses[..., 1:2], poses[..., 2:3], poses[..., 3:4]], -1)
        elif self.dataset == 'EndoNeRF':
            # load poses
            poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
            poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
            H, W, focal = poses[0, :, -1]
            focal = focal / self.downsample
            self.focal = (focal, focal)
            self.K = np.array([[focal, 0 , W//2],
                                        [0, focal, H//2],
                                        [0, 0, 1]]).astype(np.float32)
            poses = np.concatenate([poses[..., :1], -poses[..., 1:2], -poses[..., 2:3], poses[..., 3:4]], -1)
        else:
            assert 0, NotImplemented
        # prepare poses
        self.image_poses = []
        self.image_times = []
        for idx in range(poses.shape[0]):
            pose = poses[idx]
            c2w = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0) # 4x4
            
            # # ======================Generate the novel view for infer (StereoMIS)==========================
            # thetas = np.linspace(0, 30, poses.shape[0], endpoint=False)
            # c2w = update_extr(c2w, rotation_deg=thetas[idx], radii_mm=30)
            # # =================================================================================
            
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, -1] #w2c
            R = np.transpose(R) #c2w
            self.image_poses.append((R, T))
            self.image_times.append(idx / poses.shape[0])
            
        
        # get paths of images, depths, masks, etc.
        agg_fn = lambda filetype: sorted(glob.glob(os.path.join(self.root_dir, filetype, "*.png")))
        self.image_paths = agg_fn("images")
        self.depth_paths = agg_fn("depth")
        self.masks_paths = agg_fn("masks")

        assert len(self.image_paths) == poses.shape[0], "the number of images should equal to the number of poses"
        assert len(self.depth_paths) == poses.shape[0], "the number of depth images should equal to number of poses"
        assert len(self.masks_paths) == poses.shape[0], "the number of masks should equal to the number of poses"
        
    def get_caminfo(self, split):
        # break down
        # cameras = []
        if split == 'train': idxs = self.train_idxs
        elif split == 'test': idxs = self.test_idxs
        else:
            idxs = self.video_idxs
        
        cam_infos = []
        # for i in tqdm(range(len(exts))):
        for idx in tqdm(idxs):
            # mask / depth
            mask_path = self.masks_paths[idx]
            mask = Image.open(mask_path)
            #mask refer to tool are valued
            if self.dataset in ['StereoMIS']:
                mask = 255-np.array(mask) 
            elif self.dataset in ['EndoNeRF']:
                mask = np.array(mask)  
            else:
                assert 0, NotImplementedError
            assert self.tool_mask == 'use',f' for misgs,we let tool_mask be use n get all masks'
            if self.tool_mask == 'use':
                mask = 1 - np.array(mask) / 255.0
            elif self.tool_mask == 'inverse':
                mask = np.array(mask) / 255.0
            elif self.tool_mask == 'nouse':
                mask = np.ones_like(mask)
            else:
                assert 0
            assert len(mask.shape)==2
        
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path))
            close_depth = np.percentile(depth[depth!=0], 3.0)
            inf_depth = np.percentile(depth[depth!=0], 99.8)
            depth = np.clip(depth, close_depth, inf_depth) 
            depth = torch.from_numpy(depth)
            mask = self.transform(mask).bool()
            # color
            img_path = self.image_paths[idx]
            color = np.array(Image.open(self.image_paths[idx]))/255.0
            image = self.transform(color)
            # times           
            time = self.image_times[idx]
            # poses
            R, T = self.image_poses[idx]
            # fov
            FovX = focal2fov(self.focal[0], self.img_wh[0])
            FovY = focal2fov(self.focal[1], self.img_wh[1])
            
            #jj---go through misgs
            # mostimporantly: faking the required cam_metadata by misgs
            # print('todo faking/fake some required cam_metadata by misgs')
            pose = np.eye(4)
            pose[:3,:3] = R
            pose[:3,-1] = T
            cam_metadata = dict()
            cam_metadata['frame'] = image#frames[i]
            cam_metadata['cam'] = 0,#idx,#cams[i]
            cam_metadata['frame_idx'] = idx #frames_idx[i]
            cam_metadata['ego_pose'] = pose
            cam_metadata['extrinsic'] = self.K
            # cam_metadata['timestamp'] = time #cams_timestamps[i]
            cam_metadata['timestamp'] = idx #cams_timestamps[i]
            if idx in self.train_idxs:
                cam_metadata['is_val'] = False
                # self.camera_timestamps[idx]['train_timestamps'].append(time)
            else:
                cam_metadata['is_val'] = True
                # self.camera_timestamps[idx]['test_timestamps'].append(time)

            from scene.dataset_readers import CameraInfo

            # mask is used by deform3dgs
            # masks is used by misgs
            # load more other masks in addtion to mask...
            # if you want
            masks = {
                # "tissue_mask":mask,
                "tissue_mask":mask,
                "tool_mask":~mask,
                "no_mask":torch.ones_like(mask).bool().to(mask.device),
            }

            cam_info = CameraInfo(
                    R=R, 
                    T=T, 
                    FovX=FovX, 
                    FovY=FovY, 
                    K=self.K,
                    image=image, 
                    image_name=f"{idx}", 
                    metadata=cam_metadata,
                    image_path=img_path,
                    #exclusive to misgs
                    id=idx, 
                    # acc_mask
                    #exclusive to deform3dgs
                    uid=idx,
                    time=time,
                    mask=mask, 
                    depth=depth, 
                    gt_alpha_mask=None,
                    data_device=torch.device("cuda"), 
                    Znear=None, Zfar=None, 
                    h=self.img_wh[1], w=self.img_wh[0],
                    masks = masks,
                )
            cam_infos.append(cam_info)

        return cam_infos


    def format_infos(self, split):
        #new version
        # did exactily the same thing, 
        # but break down for better extending
        # mostimporantly: faking the required cam_metadata by misgs
        cam_infos = self.get_caminfo(split=split)

        from utils.camera_utils import cameraList_from_camInfos
        #we rewrite the cameralist_from_caminfo func        
        return cameraList_from_camInfos(cam_infos)

    
    def filling_pts_colors(self, filling_mask, ref_depth, ref_image):
         # bool
        refined_depth = inpaint_depth(ref_depth, filling_mask)
        refined_rgb = inpaint_rgb(ref_image, filling_mask)
        return refined_rgb, refined_depth

    
    def get_sparse_pts_dict_misgs(self, sample=True, init_mode = None):
        assert init_mode in ['MAPF','skipMAPF','rand']
        R, T = self.image_poses[0]
        depth = np.array(Image.open(self.depth_paths[0]))
        depth_mask = np.ones(depth.shape).astype(np.float32)
        close_depth = np.percentile(depth[depth!=0], 0.1)
        inf_depth = np.percentile(depth[depth!=0], 99.9)
        depth_mask[depth>inf_depth] = 0
        depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
        depth_mask[depth==0] = 0
        depth[depth_mask==0] = 0

        #use mask in init too
        mask = Image.open(self.masks_paths[0])
        #mask refer to tool are valued
        if self.dataset in ['StereoMIS']:
            mask = 255-np.array(mask) 
        elif self.dataset in ['EndoNeRF']:
            mask = np.array(mask)  
        else:
            assert 0, NotImplementedError
        if self.tool_mask == 'use':
            mask = 1 - np.array(mask) / 255.0
        elif self.tool_mask == 'inverse':
            mask = np.array(mask) / 255.0
        elif self.tool_mask == 'nouse':
            mask = np.ones_like(mask)
        else:
            assert 0
        assert len(mask.shape)==2

        masks_dict = {'tissue':mask,
                      'obj_tool1':1-mask}
        pts_dict = {}
        colors_dict = {}
        normals_dict = {}
        for piece_name,piece_mask in masks_dict.items():
            mask = np.logical_and(depth_mask, piece_mask)   
            color = np.array(Image.open(self.image_paths[0]))/255.0
            pts, colors, _ = self.get_pts_cam(depth, mask, color, disable_mask=False)
            c2w = self.get_camera_poses((R, T))
            pts = self.transform_cam2cam(pts, c2w)
            
            if init_mode=='skipMAPF':
                print('alright consider mask')
                pass
            elif init_mode == 'MAPF':
                print('alright consider mask')
                pts, colors = self.search_pts_colors_with_motion(pts, colors, mask, c2w)#MAPF
            elif init_mode == 'rand':
                #/////////////////////////////////////////
                rand_num_pts = 100_000  
                warnings.warn(f"tissue rand init(w.o concerning mask): generating random point cloud ({rand_num_pts})... w.o mask constrains?")
                # use the params from deformable-3d-gs synthetic Blender scenes
                pts = np.random.random((rand_num_pts, 3)) * 2.6 - 1.3
                shs = np.random.random((rand_num_pts, 3)) / 255.0
                colors=SH2RGB(shs)
                #//////////////////////
            else:
                assert 0, NotImplementedError
            normals = np.zeros((pts.shape[0], 3))

            if sample:
                num_sample = int(0.1 * pts.shape[0])
                sel_idxs = np.random.choice(pts.shape[0], num_sample, replace=False)
                pts = pts[sel_idxs, :]
                colors = colors[sel_idxs, :]
                normals = normals[sel_idxs, :]

            pts_dict[piece_name] = pts
            colors_dict[piece_name] = colors
            normals_dict[piece_name] = normals
        # return pts, colors, normals
        # return pts_dict,colors_dict,normals_dict
        return pts_dict,colors_dict,normals_dict, masks_dict


    def get_sparse_pts(self, sample=True, init_mode = None):
        assert init_mode in ['MAPF','skipMAPF','rand']
        R, T = self.image_poses[0]
        depth = np.array(Image.open(self.depth_paths[0]))
        depth_mask = np.ones(depth.shape).astype(np.float32)
        close_depth = np.percentile(depth[depth!=0], 0.1)
        inf_depth = np.percentile(depth[depth!=0], 99.9)
        depth_mask[depth>inf_depth] = 0
        depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
        depth_mask[depth==0] = 0
        depth[depth_mask==0] = 0

        #use mask in init too
        mask = Image.open(self.masks_paths[0])
        #mask refer to tool are valued
        if self.dataset in ['StereoMIS']:
            mask = 255-np.array(mask) 
        elif self.dataset in ['EndoNeRF']:
            mask = np.array(mask)  
        else:
            assert 0, NotImplementedError

        if self.tool_mask == 'use':
            mask = 1 - np.array(mask) / 255.0
        elif self.tool_mask == 'inverse':
            mask = np.array(mask) / 255.0
        elif self.tool_mask == 'nouse':
            mask = np.ones_like(mask)
        else:
            assert 0
        assert len(mask.shape)==2
        mask = np.logical_and(depth_mask, mask)   
        color = np.array(Image.open(self.image_paths[0]))/255.0
        pts, colors, _ = self.get_pts_cam(depth, mask, color, disable_mask=False)
        c2w = self.get_camera_poses((R, T))
        pts = self.transform_cam2cam(pts, c2w)
        
        if init_mode=='skipMAPF':
            print('alright consider mask')
            pass
        elif init_mode == 'MAPF':
            print('alright consider mask')
            pts, colors = self.search_pts_colors_with_motion(pts, colors, mask, c2w)#MAPF
        elif init_mode == 'rand':
            #/////////////////////////////////////////
            rand_num_pts = 100_000
            warnings.warn(f"tissue rand init(w.o concerning mask): generating random point cloud ({rand_num_pts})... w.o mask constrains?")
            # use the params from deformable-3d-gs synthetic Blender scenes
            pts = np.random.random((rand_num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((rand_num_pts, 3)) / 255.0
            colors=SH2RGB(shs)
            #//////////////////////
        else:
            assert 0, NotImplementedError
        normals = np.zeros((pts.shape[0], 3))

        if sample:
            num_sample = int(0.1 * pts.shape[0])
            sel_idxs = np.random.choice(pts.shape[0], num_sample, replace=False)
            pts = pts[sel_idxs, :]
            colors = colors[sel_idxs, :]
            normals = normals[sel_idxs, :]
        
        return pts, colors, normals

    def calculate_motion_masks(self, ):
        images = []
        for j in range(0, len(self.image_poses)):
            color = np.array(Image.open(self.image_paths[j]))/255.0
            images.append(color)
        images = np.asarray(images).mean(axis=-1)
        diff_map = np.abs(images - images.mean(axis=0))
        diff_thrshold = np.percentile(diff_map[diff_map!=0], 95)
        return diff_map > diff_thrshold
        
    def search_pts_colors_with_motion(self, ref_pts, ref_color, ref_mask, ref_c2w):
        # calculating the motion mask
        motion_mask = self.calculate_motion_masks()
        interval = 1
        if len(self.image_poses) > 150: # in case long sequence
            interval = 2
        for j in range(1,  len(self.image_poses), interval):
            ref_mask_not = np.logical_not(ref_mask)
            ref_mask_not = np.logical_or(ref_mask_not, motion_mask[0])
            R, T = self.image_poses[j]
            c2w = self.get_camera_poses((R, T))
            c2ref = np.linalg.inv(ref_c2w) @ c2w
            depth = np.array(Image.open(self.depth_paths[j]))
            color = np.array(Image.open(self.image_paths[j]))/255.0
            # mask = 1 - np.array(Image.open(self.masks_paths[0]))/255.0  

            mask = Image.open(self.masks_paths[j])
            #mask refer to tool are valued
            if self.dataset in ['StereoMIS']:
                mask = 255-np.array(mask) 
            elif self.dataset in ['EndoNeRF']:
                mask = np.array(mask)  
            else:
                assert 0, NotImplementedError
                
            if self.tool_mask == 'use':
                mask = 1 - np.array(mask) / 255.0
            elif self.tool_mask == 'inverse':
                mask = np.array(mask) / 255.0
            elif self.tool_mask == 'nouse':
                mask = np.ones_like(mask)
            else:
                assert 0
            assert len(mask.shape)==2
            depth_mask = np.ones(depth.shape).astype(np.float32)
            close_depth = np.percentile(depth[depth!=0], 3.0)
            inf_depth = np.percentile(depth[depth!=0], 99.8)
            depth_mask[depth>inf_depth] = 0
            depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
            depth_mask[depth==0] = 0
            mask = np.logical_and(depth_mask, mask)
            depth[mask==0] = 0
            
            pts, colors, mask_refine = self.get_pts_cam(depth, mask, color)
            pts = self.transform_cam2cam(pts, c2ref) # Nx3
            X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
            X, Y, Z = X[Z!=0], Y[Z!=0], Z[Z!=0]
            X_Z, Y_Z = X / Z, Y / Z
            X_Z = (X_Z * self.focal[0] + self.K[0,-1]).astype(np.int32)
            Y_Z = (Y_Z * self.focal[1] + self.K[1,-1]).astype(np.int32)
            # Out of the visibility
            out_vis_mask = ((X_Z > (self.img_wh[0]-1)) + (X_Z < 0) +\
                    (Y_Z > (self.img_wh[1]-1)) + (Y_Z < 0))>0
            out_vis_pt_idx = np.where(out_vis_mask)[0]
            visible_mask = (1 - out_vis_mask)>0
            X_Z = np.clip(X_Z, 0, self.img_wh[0]-1)
            Y_Z = np.clip(Y_Z, 0, self.img_wh[1]-1)
            coords = np.stack((Y_Z, X_Z), axis=-1)
            proj_mask = np.zeros((self.img_wh[1], self.img_wh[0])).astype(np.float32)
            proj_mask[coords[:, 0], coords[:, 1]] = 1
            compl_mask = (ref_mask_not * proj_mask)
            index_mask = compl_mask.reshape(-1)[mask_refine]
            compl_idxs = np.nonzero(index_mask.reshape(-1))[0]
            if compl_idxs.shape[0] <= 50:
                continue
            compl_pts = pts[compl_idxs, :]
            compl_pts = self.transform_cam2cam(compl_pts, ref_c2w)
            compl_colors = colors[compl_idxs, :]
            sel_idxs = np.random.choice(compl_pts.shape[0], int(0.1*compl_pts.shape[0]), replace=True)
            ref_pts = np.concatenate((ref_pts, compl_pts[sel_idxs]), axis=0)
            ref_color = np.concatenate((ref_color, compl_colors[sel_idxs]), axis=0)
            ref_mask = np.logical_or(ref_mask, compl_mask)

        
        if ref_pts.shape[0] > 600000:
            sel_idxs = np.random.choice(ref_pts.shape[0], 500000, replace=True)  
            ref_pts = ref_pts[sel_idxs]         
            ref_color = ref_color[sel_idxs] 
        return ref_pts, ref_color
    
         
    def get_camera_poses(self, pose_tuple):
        R, T = pose_tuple
        R = np.transpose(R)
        w2c = np.concatenate((R, T[...,None]), axis=-1)
        w2c = np.concatenate((w2c, np.array([[0, 0, 0, 1]])), axis=0)
        c2w = np.linalg.inv(w2c)
        return c2w
    
    def get_pts_cam(self, depth, mask, color, disable_mask=False):
        W, H = self.img_wh
        i, j = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H))
        cx = self.K[0,-1]
        cy = self.K[1,-1]
        X_Z = (i-cx) / self.focal[0]
        Y_Z = (j-cy) / self.focal[1]
        Z = depth
        X, Y = X_Z * Z, Y_Z * Z
        pts_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        color = color.reshape(-1, 3)
        
        if not disable_mask:
            mask = mask.reshape(-1).astype(bool)
            pts_valid = pts_cam[mask, :]
            color_valid = color[mask, :]
        else:
            mask = mask.reshape(-1).astype(bool)
            pts_valid = pts_cam
            color_valid = color
            color_valid[mask==0, :] = np.ones(3)
                    
        return pts_valid, color_valid, mask
        
    def get_maxtime(self):
        return self.maxtime
    
    def transform_cam2cam(self, pts_cam, pose):
        pts_cam_homo = np.concatenate((pts_cam, np.ones((pts_cam.shape[0], 1))), axis=-1)
        pts_wld = np.transpose(pose @ np.transpose(pts_cam_homo))
        xyz = pts_wld[:, :3]
        return xyz
    def load_other_obj_meta(self,cameras = [0],num_frames = None):
        scene_metadata = {}
        #fake:
        obj_tracklets = None
        tracklet_timestamps = None
        obj_info = None
        # num_frames = None
        obj_tracklets = None
        exts = []

        scene_metadata = dict()
        scene_metadata['obj_tracklets'] = obj_tracklets
        scene_metadata['tracklet_timestamps'] = tracklet_timestamps
        scene_metadata['obj_meta'] = obj_info
        scene_metadata['num_images'] = len(exts)
        # scene_metadata['num_cams'] = len(cfg.data.cameras)
        scene_metadata['num_cams'] = len(cameras)
        scene_metadata['num_frames'] = num_frames

        #most important for misgs
        print('most important for misgs')
        scene_metadata['camera_timestamps'] = self.camera_timestamps
 
        return scene_metadata
    

    # jj
    def get_endonerf_cam_meta(self,cameras = [0]):
        # assert 0, NotImplementedError
        metadata = {}#cam_metadata
        return metadata
        # for cam in cfg.data.get('cameras', [0, 1, 2]):
        # for cam in cameras:
        #     camera_timestamps[cam] = dict()
        #     camera_timestamps[cam]['train_timestamps'] = []
        #     camera_timestamps[cam]['test_timestamps'] = []   
        # for i in tqdm(range(len(exts))):
        #     metadata = dict()
        #     metadata['frame'] = frames[i]
        #     metadata['cam'] = cams[i]
        #     metadata['frame_idx'] = frames_idx[i]
        #     metadata['ego_pose'] = pose
        #     metadata['extrinsic'] = ext
        #     metadata['timestamp'] = cams_timestamps[i]

        #     if frames_idx[i] in train_frames:
        #         metadata['is_val'] = False
        #         camera_timestamps[cams[i]]['train_timestamps'].append(cams_timestamps[i])
        #     else:
        #         metadata['is_val'] = True
        #         camera_timestamps[cams[i]]['test_timestamps'].append(cams_timestamps[i])
            
        #     # load dynamic mask
        #     if load_dynamic_mask:
        #         # dynamic_mask_path = os.path.join(dynamic_mask_dir, f'{image_name}.png')
        #         # obj_bound = (cv2.imread(dynamic_mask_path)[..., 0]) > 0.
        #         # obj_bound = Image.fromarray(obj_bound)
        #         metadata['obj_bound'] = Image.fromarray(obj_bounds[i])
        #     #do we need cam meta?/////////////
        #     mask = None        
        #     cam_info = CameraInfo(
        #         uid=i, R=R, T=T, FovY=FovY, FovX=FovX, K=K,
        #         image=image, image_path=image_path, image_name=image_name,
        #         width=width, height=height, 
        #         mask=mask,
        #         metadata=metadata)

        #     # sys.stdout.write('\n')
