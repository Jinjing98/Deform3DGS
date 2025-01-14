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
                 cam_id = 0,
                 cfg = None,
                 tracklets = None,
                 tracklet_timestamps = None,
                 ):
        # tracklets: [num_frames, max_obj, [track_id, x, y, z, qw, qx, qy, qz]]
        # frame_timestamps: [num_frames]
        super().__init__()
        print('Think about learn with lie..(now rpy)')
        self.cfg = cfg
        self.cfg_optim = cfg_optim
        self.camera_timestamps = camera_timestamps
        self.timestamps = self.camera_timestamps[str(cam_id)]['all_timestamps']
        # we predict abs pose
        frames_num = len(self.timestamps)
        # obj_pose_rot_optim_space = 'rpy', #'lie'
        assert objs_num == 1,objs_num
        if self.cfg_optim.obj_pose_init == '0':
            self.input_trans = torch.zeros([frames_num,objs_num,3]).float().cuda()
            self.input_rots_quat = torch.zeros([frames_num,objs_num,4]).float().cuda() #wxyz
            self.input_rots_quat[:,:,0] = 1
        elif self.cfg_optim.obj_pose_init == 'cotrackerpnp':
            self.input_trans = torch.zeros([frames_num,objs_num,3]).float().cuda()
            self.input_rots_quat = torch.zeros([frames_num,objs_num,4]).float().cuda() #wxyz
            self.input_rots_quat[:,:,0] = 1
            for i in range(objs_num):
                cotrackerpnp_trajectory_cams2w = tracklets[f'obj_tool{i+1}']['trajectory_cams2w'].float().cuda()# 
                load_num,_,_ = cotrackerpnp_trajectory_cams2w.shape
                assert load_num == frames_num
                cotrackerpnp_trajectory_w2cams2 = torch.linalg.inv(cotrackerpnp_trajectory_cams2w)
                self.input_trans[:,i,:] = cotrackerpnp_trajectory_w2cams2[:,:3,3]
                input_rots_mat = cotrackerpnp_trajectory_w2cams2[:,:3,:3]
                self.input_rots_quat[:,i,:] = matrix_to_quaternion(input_rots_mat)#cotrackerpnp_trajectory_cams2w[:,:3,3]      
        else:
            assert 0,  self.cfg_optim.obj_pose_init

        self.opt_track = opt_track #cfg.model.nsg.opt_track
        if self.opt_track:
            self.opt_trans = nn.Parameter(torch.zeros_like(self.input_trans)).requires_grad_(True).to(self.input_trans.device) 
            f_num,obj_num,_  = self.opt_trans.shape
            self.opt_rots_rpy = nn.Parameter(torch.zeros([f_num,obj_num,3],
                                                         device = self.input_trans.device)).requires_grad_(True)\
                                                            .to(self.opt_trans.device).to(self.opt_trans.dtype)  
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
                                                    # max_steps=self.cfg.train.iterations,
                                                    max_steps=self.cfg_optim.track_position_max_steps,
                                                    # warmup_steps=self.cfg_optim.opacity_reset_interval,
                                                    warmup_steps=self.cfg_optim.track_warmup_steps,
                                                    )
            
            self.opt_rots_rpy_scheduler_args = get_expon_lr_func(lr_init=self.cfg_optim.track_rotation_lr_init,
                                                    lr_final=self.cfg_optim.track_rotation_lr_final,
                                                    lr_delay_mult=self.cfg_optim.track_rotation_lr_delay_mult,
                                                    # max_steps=self.cfg.train.iterations,
                                                    max_steps=self.cfg_optim.track_rotation_max_steps,
                                                    # warmup_steps=self.cfg_optim.opacity_reset_interval,
                                                    warmup_steps=self.cfg_optim.track_warmup_steps,
                                                    
                                                    ) 

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
        trans = self.opt_trans[frame_idx, track_id] 
        # print(f'debug opt_trans {frame_idx}: all_0?{not trans.any()} {trans}')
        # if frame_idx == 24:
            # pass
        # print(f"debug ***********opt_trans{len(self.opt_trans)}",frame_idx,"delta",self.opt_trans[-3:],
            #   "input",self.input_trans[-3:])
        trans = trans + self.input_trans[frame_idx, track_id] 
        return trans


   
    def get_tracking_rotation(self, track_id, camera: Camera):
        '''
        param to learn is rpy
        return in wxyz format(gs)'''
        
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

        def euler_to_quaternion_torch(roll_pitch_yaw):
            """
            Convert Euler angles (roll, pitch, yaw) to a quaternion (w, x, y, z).
            Roll, pitch, and yaw should be in radians.
            """
            assert roll_pitch_yaw.shape == torch.Size([3])
            roll,pitch,yaw = roll_pitch_yaw
            # Compute half angles
            cy = torch.cos(yaw * 0.5)
            sy = torch.sin(yaw * 0.5)
            cp = torch.cos(pitch * 0.5)
            sp = torch.sin(pitch * 0.5)
            cr = torch.cos(roll * 0.5)
            sr = torch.sin(roll * 0.5)

            # Compute quaternion
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy

            # quant = torch.zeros_like(torch.Tensor([w,x,y,z])).to(roll_pitch_yaw.device)
            # quant.requires_grad = True
            # quant[0:4] = torch.Tensor([w,x,y,z]).to(roll_pitch_yaw.device)
            # return quant
            return torch.stack((w, x, y, z), dim=-1)
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
        # quaternion = euler_to_quaternion(roll_pitch_yaw)
        quaternion = euler_to_quaternion_torch(roll_pitch_yaw)

        # roll_pitch_yaw_input = self.input_rots_rpy[frame_idx,track_id]
        # quaternion_input = euler_to_quaternion(roll_pitch_yaw_input)
        quaternion_input = self.input_rots_quat[frame_idx,track_id]
        # if frame_idx == 24:
            # print("debug ***********opt_rots_rpy",frame_idx,"delta",quaternion.data,"input",quaternion_input.data)
        quaternion = quaternion_raw_multiply(quaternion_input.unsqueeze(0), 
                                    quaternion.unsqueeze(0)).squeeze(0)
        # print("Quaternion (w, x, y, z):", quaternion)
        return quaternion

    def save_state_dict(self, 
                        is_final = False,
                        ):
        state_dict = dict()
        if self.opt_track:
            state_dict['params'] = self.state_dict()
        if not is_final:
            state_dict['optimizer'] = self.optimizer.state_dict()
        return state_dict
