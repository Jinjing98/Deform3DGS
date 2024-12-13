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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.tool_model import ToolModel
from utils.sh_utils import eval_sh


 
def tool_render(viewpoint_camera, pc : ToolModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # try:
    #     screenspace_points.retain_grad()
    # except:
    #     pass

    #code from deform gs : jinjing
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_densify = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_densify.retain_grad()
    except:
        pass


    # Set up rasterization configuration
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation
    means3D = pc.get_xyz
    ori_time = torch.tensor(viewpoint_camera.time).to(means3D.device)
    means2D = screenspace_points
    opacity = pc._opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    # deformation_point = pc._deformation_table
    # means3D_deform, scales_deform, rotations_deform = pc.deformation(means3D[deformation_point], scales[deformation_point], 
                                                                        #  rotations[deformation_point],
                                                                        #  ori_time)
    # opacity_deform = opacity[deformation_point]
        
    # print(time.max())
    # with torch.no_grad():
        # pc._deformation_accum[deformation_point] += torch.abs(means3D_deform - means3D[deformation_point])

    # means3D_final = torch.zeros_like(means3D)
    # rotations_final = torch.zeros_like(rotations)
    # scales_final = torch.zeros_like(scales)
    # opacity_final = torch.zeros_like(opacity)
    # means3D_final[deformation_point] =  means3D_deform
    # rotations_final[deformation_point] =  rotations_deform
    # scales_final[deformation_point] =  scales_deform
    # opacity_final[deformation_point] = opacity_deform
    # means3D_final[~deformation_point] = means3D[~deformation_point]
    # rotations_final[~deformation_point] = rotations[~deformation_point]
    # scales_final[~deformation_point] = scales[~deformation_point]
    # opacity_final[~deformation_point] = opacity[~deformation_point]


    means3D_final = means3D
    rotations_final = rotations
    scales_final = scales
    opacity_final = opacity







    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity_final = pc.opacity_activation(opacity_final)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
 
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii, depth = rasterizer(
    # rendered_image, radii, depth, _, _ = rasterizer( #latest: no means2d_densify; return 5 values
    rendered_image, radii, depth,  = rasterizer( #latest: no means2d_densify; return 5 values
        means3D = means3D_final,
        means2D = means2D,
        #jj
        # means2D_densify=screenspace_points_densify,
        shs = shs,
        colors_precomp = colors_precomp,
        # opacities = opacity,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    # rendered_image_vis_tool = rendered_image.detach().to('cpu')
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,}
