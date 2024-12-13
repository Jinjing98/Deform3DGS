import torch
import torch.nn as nn
import numpy as np
from utils.general_utils import quaternion_raw_multiply, get_expon_lr_func, quaternion_slerp, matrix_to_quaternion
from utils.camera_utils import Camera

class ToolPose(nn.Module):      
    def __init__(self, 
                #  frames_num,
                 objs_num = 1,
                 camera_timestamps = None, 
                #  obj_info, 
                 cfg_optim = None,
                 opt_track = True,
                 cam_id = 0):
        # tracklets: [num_frames, max_obj, [track_id, x, y, z, qw, qx, qy, qz]]
        # frame_timestamps: [num_frames]
        super().__init__()

        self.cfg_optim = cfg_optim
        self.camera_timestamps = camera_timestamps
        self.timestamps = self.camera_timestamps[str(cam_id)]['all_timestamps']
        # we predict abs pose
        frames_num = len(self.timestamps)
        self.input_trans = torch.zeros([frames_num,objs_num,3]).float().cuda()
        # self.input_rots_mat = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(frames_num,objs_num,1,1).float().cuda()
        self.input_rots_rpy = torch.zeros([frames_num,objs_num,3]).float().cuda()
        # self.input_rots_quant = torch.zeros([frames_num,objs_num,3]).float().cuda()
        assert objs_num == 1,objs_num

        self.opt_track = opt_track #cfg.model.nsg.opt_track
        if self.opt_track:
            self.opt_trans = nn.Parameter(torch.zeros_like(self.input_trans)).requires_grad_(True) 
            self.opt_rots_rpy = nn.Parameter(torch.zeros_like(self.input_rots_rpy)).requires_grad_(True) 
        else:
            assert 0, NotImplementedError


    def training_setup(self):
        if self.opt_track:
            params = [
                {'params': [self.opt_trans], 'lr': self.cfg_optim.track_position_lr_init, 'name': 'opt_trans'},
                {'params': [self.opt_rots_rpy], 'lr': self.cfg_optim.track_rotation_lr_init, 'name': 'opt_rots_rpy'},
                # {'params': [self.opt_rots_mat], 'lr': self.cfg_optim.track_rotation_lr_init, 'name': 'opt_rots_mat'},
            ]
            
            self.opt_trans_scheduler_args = get_expon_lr_func(lr_init=self.cfg_optim.track_position_lr_init,
                                                    lr_final=self.cfg_optim.track_position_lr_final,
                                                    lr_delay_mult=self.cfg_optim.track_position_lr_delay_mult,
                                                    max_steps=self.cfg_optim.track_position_max_steps,
                                                    warmup_steps=self.cfg_optim.opacity_reset_interval)
            
            self.opt_rots_rpy_scheduler_args = get_expon_lr_func(lr_init=self.cfg_optim.track_rotation_lr_init,
                                                    lr_final=self.cfg_optim.track_rotation_lr_final,
                                                    lr_delay_mult=self.cfg_optim.track_rotation_lr_delay_mult,
                                                    max_steps=self.cfg_optim.track_rotation_max_steps,
                                                    warmup_steps=self.cfg_optim.opacity_reset_interval) 

            # self.opt_rots_mat_scheduler_args = get_expon_lr_func(lr_init=self.cfg_optim.track_rotation_lr_init,
            #                                         lr_final=self.cfg_optim.track_rotation_lr_final,
            #                                         lr_delay_mult=self.cfg_optim.track_rotation_lr_delay_mult,
            #                                         max_steps=self.cfg_optim.track_rotation_max_steps,
            #                                         warmup_steps=self.cfg_optim.opacity_reset_interval)    
            
            self.optimizer = torch.optim.Adam(params=params, lr=0, eps=1e-15)
        else:
            assert 0
    
    def update_learning_rate(self, iteration):
        if self.opt_track:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "opt_trans":
                    lr = self.opt_trans_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "opt_rots_rpy":
                    lr = self.opt_rots_rpy_scheduler_args(iteration)
                    param_group['lr'] = lr
                # if param_group["name"] == "opt_rots_mat":
                #     lr = self.opt_rots_mat_scheduler_args(iteration)
                #     param_group['lr'] = lr
        
    def update_optimizer(self):
        if self.opt_track:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=None)
        

    

    def get_tracking_translation(self, track_id, camera: Camera):
        assert track_id==0
        cam_timestamp = camera.meta['timestamp']
        frame_idx = self.timestamps.index(cam_timestamp)
        # return self.input_trans[frame_idx, track_id]
        return self.opt_trans[frame_idx, track_id]


   
    def get_tracking_rotation(self, track_id, camera: Camera):
        '''
        param to learn is rpy
        return in wxyz format(gs)'''
        
        print('Think about learn with lie..(now rpy)')
        import math
        def euler_to_quaternion(roll_pitch_yaw):
            """
            Convert Euler angles (roll, pitch, yaw) to a quaternion (w, x, y, z).
            Roll, pitch, and yaw should be in radians.
            """
            assert roll_pitch_yaw.shape == torch.Size([3])
            roll,pitch,yaw = roll_pitch_yaw
            # Compute half angles
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            cp = math.cos(pitch * 0.5)
            sp = math.sin(pitch * 0.5)
            cr = math.cos(roll * 0.5)
            sr = math.sin(roll * 0.5)

            # Compute quaternion
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy

            return torch.Tensor([w,x,y,z]).to(roll_pitch_yaw.device)
            return w, x, y, z


        # Example usage:
        # roll = 0.1  # radians
        # pitch = 0.2  # radians
        # yaw = 0.3  # radians
        assert track_id==0

        cam_timestamp = camera.meta['timestamp']
        frame_idx = self.timestamps.index(cam_timestamp)
        roll_pitch_yaw = self.opt_rots_rpy[frame_idx,track_id]
        quaternion = euler_to_quaternion(roll_pitch_yaw)
        print("Quaternion (w, x, y, z):", quaternion)
        return quaternion

        




