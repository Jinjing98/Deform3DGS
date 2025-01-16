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
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def TV_loss(x, mask):
    B, C, H, W = x.shape
    tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
    tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
    return (tv_h + tv_w) / (B * C * H * W)


def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()

def l1_loss(network_output, gt, mask=None):
    assert network_output.ndim in [3,4]
    assert gt.ndim in [3,4]
    if network_output.ndim==3:
        network_output = network_output.unsqueeze(0)
    if gt.ndim==3:
        gt = gt.unsqueeze(0)
    assert network_output.ndim==gt.ndim,f"{network_output.shape} {gt.shape}"

    if mask!= None:
        assert mask.ndim in [2,3,4]
        if mask.ndim==2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim==3:
            mask = mask.unsqueeze(0)
        assert network_output.ndim==mask.ndim

    # ///////////////////////////
    loss = torch.abs(network_output - gt)
    if mask is not None:
        if mask.ndim == 4:
            assert network_output.ndim == 4
            mask = mask.repeat(1, network_output.shape[1], 1, 1)
        else:
            # raise ValueError('the dimension of mask should be either 3 or 4')
            raise ValueError(f'the dimension of mask should be either 3 or 4 \
                             {mask.shape} {gt.shape} {network_output.shape} {loss.shape}')
    
        try:
            loss = loss[mask!=0]
        except:
            print(loss.shape)
            print(mask.shape)
            print(loss.dtype)
            print(mask.dtype)
            assert 0,loss.mean()
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

# streegs added
def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if mask is not None:
        img1 = torch.where(mask, img1, torch.zeros_like(img1))
        img2 = torch.where(mask, img2, torch.zeros_like(img2))
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)





if __name__ == "__main__":
    pass
    # from pytorch3d.io import load_ply
    # from pytorch3d.ops import ball_query
    # import pickle
    # with open("./control_kpt.pkl", "rb") as f:
    #     data = pickle.load(f)

    # points = data["pts"]
    # handle_idx = data["handle_idx"]
    # handle_pos = data["handle_pos"]

    # import trimesh
    # trimesh.Trimesh(vertices=points).export('deformation_before.ply')

    # #### prepare data
    # points = torch.from_numpy(points).float().cuda()
    # handle_idx = torch.tensor(handle_idx).long().cuda()
    # handle_pos = torch.from_numpy(handle_pos).float().cuda()

    # deformer = ARAPDeformer(points)

    # with torch.no_grad():
    #     points_prime, p_prime_seq = deformer.deform(handle_idx, handle_pos)

    # trimesh.Trimesh(vertices=points_prime.cpu().numpy()).export('deformation_after.ply')

    # from utils.deform_utils import cal_arap_error
    # for p_prime in p_prime_seq:
    #     nodes_sequence = torch.cat([points[None], p_prime[None]], dim=0)
    #     arap_error = cal_arap_error(nodes_sequence, deformer.ii, deformer.jj, deformer.nn, K=deformer.K, weight=deformer.normalized_weight)
    #     print(arap_error)



    # arap_loss(self, t=None, delta_t=0.05, t_samp_num=2):
        # t = torch.rand([]).cuda() if t is None else t.squeeze() + delta_t * (torch.rand([]).cuda() - .5)
        # t_samp = torch.rand(t_samp_num).cuda() * delta_t + t - .5 * delta_t
        # # M 512?
        # t_samp = t_samp[None, :, None].expand(self.node_num, t_samp_num, 1)  # M, T, 1
        # node_trans = self.node_deform(t=t_samp)['d_xyz']
        # nodes_t = self.nodes[:, None, :3].detach() + node_trans  # M, T, 3
        # hyper_nodes = nodes_t[:,0]  # M, 3
        # # K constrain the nbr area
        # ii, jj, nn, weight = cal_connectivity_from_points(hyper_nodes, K=10)  # connectivity of control nodes
        # error = cal_arap_error(nodes_t.permute(1,0,2), ii, jj, nn)
        # return error


    # import sys
    # sys.path.append('/mnt/ceph/tco/TCO-Staff/Homes/jinjing/proj/gs/baselines/SC-GS/utils')
    # from deform_utils import cal_connectivity_from_points,cal_arap_error
    # # from utils.deform_utils import cal_connectivity_from_points, cal_arap_error, arap_deformation_loss