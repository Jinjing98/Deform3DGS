from utils.stereo_rectify import StereoRectifier
from submodules.RAFT.core.raft import RAFT
from argparse import ArgumentParser, Action
from torchvision.transforms import Resize, InterpolationMode
from collections import OrderedDict
import os
import numpy as np
import torch
import cv2
import shutil

RAFT_config = {
    "pretrained": "/mnt/ceph/tco/TCO-Staff/Homes/jinjing/proj/endomotion/pretrained_models/raft-things.pth",
    "iters": 12,
    "dropout": 0.0,
    "small": False,
    "pose_scale": 1.0,
    "lbgfs_iters": 100,
    "use_weights": True,
    "dbg": False
}

def check_arg_limits(arg_name, n):
    class CheckArgLimits(Action):
        def __call__(self, parser, args, values, option_string=None):
            if len(values) > n:
                parser.error("Too many arguments for " + arg_name + ". Maximum is {0}.".format(n))
            if len(values) < n:
                parser.error("Too few arguments for " + arg_name + ". Minimum is {0}.".format(n))
            setattr(args, self.dest, values)
    return CheckArgLimits

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    mask = torch.from_numpy(mask).unsqueeze(0)
    return mask


class DepthEstimator(torch.nn.Module):
    def __init__(self, config):
        super(DepthEstimator, self).__init__()
        self.model = RAFT(config).to('cuda')
        self.model.freeze_bn()
        new_state_dict = OrderedDict()
        raft_ckp = config['pretrained']
        try:
            state_dict = torch.load(raft_ckp)
        except RuntimeError:
            state_dict = torch.load(raft_ckp, map_location='cpu')
        for k, v in state_dict.items():
            name = k.replace('module.','')  # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        
    def forward(self, imagel, imager, baseline, upsample=True):
        n, _, h, w = imagel.shape
        flow = self.model(imagel.to('cuda'), imager.to('cuda'), upsample=upsample)[0][-1]
        baseline = torch.from_numpy(baseline).to('cuda')
        depth = baseline[:, None, None] / -flow[:, 0]
        if not upsample:
            depth/= 8.0  # factor 8 of upsampling
        valid = (depth > 0) & (depth <= 250.0)
        depth[~valid] = 0.0
        return depth.unsqueeze(1)
        

def fake_endonerf_complete_depth(data_dir, intri_dir, start_frame, end_frame, \
                                 img_size=(512, 640), dst_dir_base = None, skip_depth_infer = False,
                                 ):
    """
    use SM intri due to lack of intri
    """
    if dst_dir_base == None:
        assert 0, "better write the processed data to another dir"
    else:
        #udpate
        assert 'P' in intri_dir,intri_dir
        intri_dir = os.path.join(STEREOMIS_BASE,intri_dir)
        dst_dir = dst_dir_base #os.path.join(dst_dir_base,os.path.basename(intri_dir))
        os.makedirs(dst_dir, exist_ok=True)

    # Load parameters after rectification
    calib_file = os.path.join(intri_dir, 'StereoCalibration.ini')
    assert os.path.exists(calib_file), "Calibration file not found."
    rect = StereoRectifier(calib_file, img_size_new=(img_size[1], img_size[0]), mode='conventional')
    calib = rect.get_rectified_calib()
    baseline = calib['bf'].astype(np.float32)
    intrinsics = calib['intrinsics']['left'].astype(np.float32)
    
    # Sort images and masks according to the start and end frame indexes
    data_dir = os.path.join(ENDONERF_BASE,data_dir)
    frames = sorted(os.listdir(os.path.join(data_dir, 'images')))
    # assert 0, frames
    # frames = [f for f in frames if 'l.png' in f and int(f.split('l.')[0]) >= start_frame and int(f.split('l.')[0]) <= end_frame]
    assert len(frames) > 0, "No frames found."
    resize = Resize(img_size)
    resize_msk = Resize(img_size, interpolation=InterpolationMode.NEAREST)
    
    if not skip_depth_infer:
        # Configurate depth estimator. We follow the settings of RAFT in robust-pose-estimator(https://github.com/aimi-lab/robust-pose-estimator)
        depth_estimator = DepthEstimator(RAFT_config)
    else:
        print('Read depth from local.')
    # Create folders
    # output_dir = os.path.join(data_dir, 'stereo_'+ os.path.basename(data_dir)+'_'+str(start_frame)+'_'+str(end_frame))
    output_dir = os.path.join(dst_dir, os.path.basename(data_dir)+'_'+str(start_frame)+'_'+str(end_frame))
    # image_dir = os.path.join(output_dir, 'images')
    # mask_dir = os.path.join(output_dir, 'masks')
    depth_dir = os.path.join(output_dir, 'depth')
    # if not os.path.exists(image_dir):
    #     os.makedirs(image_dir)
    # if not os.path.exists(mask_dir):
    #     os.makedirs(mask_dir)
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)
    poses_bounds = []
    from tqdm import tqdm
    print('!*******************************************')
    print('!!!!Hard set 0 pose----only static cam!')
    for i, frame in enumerate(tqdm(frames)):
        left_img = torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(data_dir, 'images', frame)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        right_img = torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(data_dir, 'images_right', frame.replace('frame-', '').replace('.color', ''))), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        left_img = resize(left_img)
        right_img = resize(right_img)
        if skip_depth_infer:
            assert 0,'not properly saved as actual depth values....check out the slam generated /flow_depth, might be proper...'
        else:
            with torch.no_grad():
                depth = depth_estimator(left_img[None], right_img[None], baseline[None])
        try:
            mask = read_mask(os.path.join(data_dir, 'masks', frame))
            mask = resize_msk(mask)
        except:
            mask = torch.ones(1, img_size[0], img_size[1])
            
        # Save the data. Of note, the file should start with 'stereo_' to be compatible with the dataloader in Deform3DGS.
        left_img_np = left_img.permute(1, 2, 0).numpy()
        mask_np = mask.permute(1, 2, 0).numpy()
        left_img_bgr = cv2.cvtColor(left_img_np, cv2.COLOR_RGB2BGR)
 
        # Save left_img, right_img, and mask to output_dir
        name = 'frame-'+str(i).zfill(6)+'.color.png'
        # cv2.imwrite(os.path.join(image_dir, name), left_img_bgr)
        # cv2.imwrite(os.path.join(mask_dir, name.replace('color','mask')), mask_np.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(depth_dir, name.replace('color','depth')), depth[0, 0].cpu().numpy())

        # Save poses_bounds.npy. Only static view is considered, i.e., R = I and T = 0.
        R = np.eye(3)
        T = np.zeros(3)
        extr = np.concatenate([R, T[:, None]], axis=1)
        cy, cx, focal = intrinsics[1, 2], intrinsics[0, 2], intrinsics[0, 0]
        param = np.concatenate([extr, np.array([[cy, cx, focal]]).T], axis=1)    
        param = param.reshape(1, 15)
        param = np.concatenate([param, np.array([[0.03, 250.]])], axis=1)
        poses_bounds.append(param[0])
        
    np.save(os.path.join(output_dir, 'poses_bounds.npy'), np.array(poses_bounds))
 

def reformat_dataset(data_dir, start_frame, end_frame, img_size=(512, 640), dst_dir_base = None, skip_depth_infer = False):
    """
    Reformat the StereoMIS to the same format as EndoNeRF dataset by stereo depth estimation.
    """
    if dst_dir_base == None:
        assert 0, "better write the processed data to another dir"
    else:
        #udpate
        assert 'P' in data_dir,data_dir
        data_dir = os.path.join(STEREOMIS_BASE,data_dir)
        dst_dir = dst_dir_base #os.path.join(dst_dir_base,os.path.basename(data_dir))
        os.makedirs(dst_dir, exist_ok=True)

    # Load parameters after rectification
    calib_file = os.path.join(data_dir, 'StereoCalibration.ini')
    assert os.path.exists(calib_file), "Calibration file not found."
    rect = StereoRectifier(calib_file, img_size_new=(img_size[1], img_size[0]), mode='conventional')
    calib = rect.get_rectified_calib()
    baseline = calib['bf'].astype(np.float32)
    intrinsics = calib['intrinsics']['left'].astype(np.float32)
    
    # Sort images and masks according to the start and end frame indexes
    frames = sorted(os.listdir(os.path.join(data_dir, 'masks')))
    frames = [f for f in frames if 'l.png' in f and int(f.split('l.')[0]) >= start_frame and int(f.split('l.')[0]) <= end_frame]
    assert len(frames) > 0, "No frames found."
    resize = Resize(img_size)
    resize_msk = Resize(img_size, interpolation=InterpolationMode.NEAREST)
    
    if not skip_depth_infer:
        # Configurate depth estimator. We follow the settings of RAFT in robust-pose-estimator(https://github.com/aimi-lab/robust-pose-estimator)
        depth_estimator = DepthEstimator(RAFT_config)
    else:
        print('Read depth from local.')
    # Create folders
    # output_dir = os.path.join(data_dir, 'stereo_'+ os.path.basename(data_dir)+'_'+str(start_frame)+'_'+str(end_frame))
    output_dir = os.path.join(dst_dir, os.path.basename(data_dir)+'_'+str(start_frame)+'_'+str(end_frame))
    image_dir = os.path.join(output_dir, 'images')
    mask_dir = os.path.join(output_dir, 'masks')
    depth_dir = os.path.join(output_dir, 'depth')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)
    poses_bounds = []
    from tqdm import tqdm
    print('!*******************************************')
    print('!!!!Hard set 0 pose----only static cam!')
    for i, frame in enumerate(tqdm(frames)):
        left_img = torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(data_dir, 'video_frames', frame)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        right_img = torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(data_dir, 'video_frames', frame.replace('l', 'r'))), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        left_img = resize(left_img)
        right_img = resize(right_img)
        if skip_depth_infer:
            assert 0,'not properly saved as actual depth values....check out the slam generated /flow_depth, might be proper...'
        else:
            with torch.no_grad():
                depth = depth_estimator(left_img[None], right_img[None], baseline[None])
        try:
            mask = read_mask(os.path.join(data_dir, 'masks', frame))
            mask = resize_msk(mask)
        except:
            mask = torch.ones(1, img_size[0], img_size[1])
            
        # Save the data. Of note, the file should start with 'stereo_' to be compatible with the dataloader in Deform3DGS.
        left_img_np = left_img.permute(1, 2, 0).numpy()
        mask_np = mask.permute(1, 2, 0).numpy()
        left_img_bgr = cv2.cvtColor(left_img_np, cv2.COLOR_RGB2BGR)
 
        # Save left_img, right_img, and mask to output_dir
        name = 'frame-'+str(i).zfill(6)+'.color.png'
        cv2.imwrite(os.path.join(image_dir, name), left_img_bgr)
        cv2.imwrite(os.path.join(mask_dir, name.replace('color','mask')), mask_np.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(depth_dir, name.replace('color','depth')), depth[0, 0].cpu().numpy())

        # Save poses_bounds.npy. Only static view is considered, i.e., R = I and T = 0.
        R = np.eye(3)
        T = np.zeros(3)
        extr = np.concatenate([R, T[:, None]], axis=1)
        cy, cx, focal = intrinsics[1, 2], intrinsics[0, 2], intrinsics[0, 0]
        param = np.concatenate([extr, np.array([[cy, cx, focal]]).T], axis=1)    
        param = param.reshape(1, 15)
        param = np.concatenate([param, np.array([[0.03, 250.]])], axis=1)
        poses_bounds.append(param[0])
        
    np.save(os.path.join(output_dir, 'poses_bounds.npy'), np.array(poses_bounds))
    
if __name__ == "__main__":            
    ENDONERF_BASE = '/mnt/cluster/datasets/EndoNeRF'
    STEREOMIS_BASE = '/mnt/ceph/tco/TCO-All/SharedDatasets/StereoMIS'
    torch.manual_seed(1234)
    np.random.seed(1234)
    # Set up command line argument parser
    parser = ArgumentParser(description="parameters for dataset format conversions")
    parser.add_argument('--dst_dir_base', '-b', type=str, default='/mnt/cluster/datasets/StereoMIS_processed')
    parser.add_argument('--data_dir', '-d', type=str, default='P2_3')
    # Frame ID of the start and end of the sequence. Of note, only 2 arguments (start and end) are required.
    parser.add_argument('--frame_id', '-f',nargs="+", action=check_arg_limits('frame_id', 2), type=int, default=[9100, 9467])
    args = parser.parse_args()
    frame_id = args.frame_id
    reformat_dataset(args.data_dir,frame_id[0], frame_id[1],dst_dir_base=args.dst_dir_base, skip_depth_infer = False)

    # data_dir = 'pulling'
    # data_dir = 'cutting'
    # fake_endonerf_complete_depth(data_dir=data_dir,
    #                              intri_dir = 'P2_3',\
    #                                 start_frame = frame_id[0], \
    #                                     end_frame=frame_id[1],\
    #                                     dst_dir_base=f'{ENDONERF_BASE}/processed', \
    #                                         skip_depth_infer = False)