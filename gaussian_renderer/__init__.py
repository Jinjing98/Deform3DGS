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
from scene.flexible_deform_model import TissueGaussianModel
from scene.tool_model import ToolModel
from utils.sh_utils import eval_sh
from typing import Union
import numpy as np

def get_final_attr_tissue(pc,viewpoint_camera_time, initial_scales,initial_opacity):
    #udpate means_3d(xyz) and rotations + other attri based on FDM
    scales = initial_scales
    opacity = initial_opacity
    means3D = pc.get_xyz
    rotations = pc._rotation
    #ori_time = torch.tensor(viewpoint_camera.time).to(means3D.device)
    ori_time = torch.tensor(viewpoint_camera_time).to(means3D.device)
    deformation_point = pc._deformation_table
    means3D_deform, scales_deform, rotations_deform = pc.deformation(means3D[deformation_point], 
                                                                    scales[deformation_point], 
                                                                    rotations[deformation_point],
                                                                    ori_time)
    opacity_deform = opacity[deformation_point]
    with torch.no_grad():
        pc._deformation_accum[deformation_point] += torch.abs(means3D_deform - means3D[deformation_point])
    #FDM
    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    opacity_final = torch.zeros_like(opacity)
    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    opacity_final[deformation_point] = opacity_deform
    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]
    opacity_final[~deformation_point] = opacity[~deformation_point]
    
    return means3D_final,rotations_final,scales_final,opacity_final
        
        
def get_final_attr_tool(misgs_model,viewpoint_camera,tool_parse_cam_again):
    #udpate means_3d(xyz) and rotations
    include_list = list(set(misgs_model.model_name_id.keys()))
    misgs_model.set_visibility(include_list)# set the self.include_list for misgs_model
    if tool_parse_cam_again:
        misgs_model.parse_camera(viewpoint_camera)# set the obj_rots/ graph_obj_list for misgs_model
    means3D_final = misgs_model.get_xyz_obj_only
    rotations_final = misgs_model.get_rotation_obj_only
    return means3D_final,rotations_final



def render_flow(viewpoint_camera,
                 pc : Union[TissueGaussianModel,ToolModel], 
                 pipe,
                 bg_color : torch.Tensor, 
                 scaling_modifier = 1.0, 
                 override_color = None,
                 debug_getxyz_misgs = False,
                 misgs_model = None,
                 single_compo_or_list = 'tissue',
                 tool_parse_cam_again = True,
                 ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if isinstance(single_compo_or_list,list):
        assert pc == None
        for name in single_compo_or_list:
            assert name in ['tissue','obj_tool1']
    elif single_compo_or_list in ['tool']:
        assert isinstance(pc,ToolModel) 
    elif single_compo_or_list in ['tissue']:
        assert isinstance(pc,TissueGaussianModel)
    else:
        assert 0,single_compo_or_list

    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # if isinstance(pc,list):
    if isinstance(single_compo_or_list,list):
        assert misgs_model != None
        assert isinstance(single_compo_or_list,list)
        # screenspace_points_list = []
        # means2D_list = []
        opacity_list = []
        scales_list = []
        sh_degree_list = []#jj add to get rid of pc
        shs_feature_list = [] # jj add to get rid of pc
        
        single_compo_or_list_idx = {}
        tgt_rendered_gs_idx = 0
        for gs_compo_name in single_compo_or_list:
            pc_i = getattr(misgs_model,gs_compo_name)
            #record the index of the compo
            single_compo_or_list_idx[gs_compo_name] = [tgt_rendered_gs_idx, tgt_rendered_gs_idx+pc_i.get_xyz.shape[0]-1]
            tgt_rendered_gs_idx += pc_i.get_xyz.shape[0]
            # we did this outside of the loop
            # get one single screenspace points outside the loop
            # screenspace_points = torch.zeros_like(pc_i.get_xyz, dtype=pc_i.get_xyz.dtype, requires_grad=True, device="cuda") + 0
            # try:
            #     screenspace_points.retain_grad()
            # except:
            #     assert 0
            #     pass
            # means2D = screenspace_points
            opacity = pc_i._opacity
            scales = pc_i._scaling
            sh_degree = pc_i.active_sh_degree#jj
            shs_feature = pc_i.get_features #jj
            
            # screenspace_points_list.append(screenspace_points)
            # means2D_list.append(means2D)
            # screenspace_points_list.append(screenspace_points_i)
            # means2D_list.append(means2D_i)
            opacity_list.append(opacity)
            scales_list.append(scales)
            sh_degree_list.append(sh_degree)#jj
            shs_feature_list.append(shs_feature)#jj
        
        #stack in the order of single_compo_or_list
        # screenspace_points = torch.vstack(screenspace_points_list)
        # means2D = torch.vstack(means2D_list)

        # init screenspace_points and means2D for saving grad in NDC space
        total_gs_num = torch.vstack(scales_list).shape[0]
        screenspace_points = torch.zeros((total_gs_num,3), dtype=pc_i.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2D = screenspace_points
        opacity = torch.vstack(opacity_list)
        scales = torch.vstack(scales_list)

        assert len(np.unique(sh_degree_list))==1,'tisseu and tool sh_degree are both 0 for each compo pc'
        sh_degree = np.unique(sh_degree_list)[0]#jj
        shs_feature = torch.vstack(shs_feature_list)#jj

        # the activation for tool and tisseu are the same-so we can use the last pc_i to getn the activation
        pc_scaling_activation = pc_i.scaling_activation
        pc_rotation_activation = pc_i.rotation_activation
        pc_opacity_activation = pc_i.opacity_activation
            
    else:
        single_compo_or_list_idx = None

        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2D = screenspace_points
        opacity = pc._opacity
        scales = pc._scaling
        sh_degree = pc.active_sh_degree
        shs_feature = pc.get_features#jj
        
        pc_scaling_activation = pc.scaling_activation
        pc_rotation_activation = pc.rotation_activation
        pc_opacity_activation = pc.opacity_activation
    # pc done here
    # print('pc done here.........')
    
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
        # sh_degree=pc.active_sh_degree,
        sh_degree=sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    if pipe.compute_cov3D_python:
        assert 0,'todo still need pc'
        # scales = None
        # rotations = None
        # cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        cov3D_precomp = None
    #//////////////fdm
    if single_compo_or_list == 'tissue':
        assert isinstance(pc,TissueGaussianModel) 
        # sh_degree = pc.active_sh_degree,
        # print('debug*************tissue*',pc.active_sh_degree)
        means3D_final,rotations_final,scales_final,opacity_final = get_final_attr_tissue(pc = pc,
                                                                                         viewpoint_camera_time = viewpoint_camera.time,
                                                                                         initial_scales = scales,
                                                                                         initial_opacity= opacity,
                                                                                         )
    elif single_compo_or_list == 'tool':
        assert isinstance(pc,ToolModel) 
        # print('debug*************tool*',pc.active_sh_degree)
        assert debug_getxyz_misgs
        assert misgs_model != None
        means3D_final,rotations_final = get_final_attr_tool(misgs_model=misgs_model,
                                                            viewpoint_camera=viewpoint_camera,
                                                            tool_parse_cam_again = tool_parse_cam_again)
        scales_final = scales
        opacity_final = opacity
    elif isinstance(single_compo_or_list,list):

        # assert debug_getxyz_misgs
        assert misgs_model != None
        means3D_final_list = []
        rotations_final_list = []
        scales_final_list = []
        opacity_final_list = []

        for gs_compo_name in single_compo_or_list:
            # notice have to follow the order of single_compo_or_list as done for means2D
            pc_i = getattr(misgs_model,gs_compo_name)
            start_idx, end_idx = single_compo_or_list_idx[gs_compo_name]

            scales_i = scales[start_idx:(end_idx+1)]
            opacity_i = opacity[start_idx:(end_idx+1)]
            if isinstance(pc_i,ToolModel):
                # print('debug /// render_flow viewpoint_camera',viewpoint_camera.id)
                means3D_final,rotations_final = get_final_attr_tool(misgs_model=misgs_model,
                                                                    viewpoint_camera=viewpoint_camera,
                                                                    tool_parse_cam_again = tool_parse_cam_again)
                scales_final = scales_i
                opacity_final = opacity_i
            elif isinstance(pc_i,TissueGaussianModel):
                means3D_final,rotations_final,scales_final,opacity_final = get_final_attr_tissue(pc = pc_i,
                                                                                                 viewpoint_camera_time = viewpoint_camera.time,
                                                                                                 initial_scales = scales_i,
                                                                                                 initial_opacity= opacity_i,
                                                                                                 )
            else:
                assert 0
            means3D_final_list.append(means3D_final)
            rotations_final_list.append(rotations_final)
            scales_final_list.append(scales_final)
            opacity_final_list.append(opacity_final)

        means3D_final = torch.vstack(means3D_final_list)
        rotations_final = torch.vstack(rotations_final_list)
        scales_final = torch.vstack(scales_final_list)
        opacity_final = torch.vstack(opacity_final_list)
    else:
        assert 0

    scales_final = pc_scaling_activation(scales_final)
    rotations_final = pc_rotation_activation(rotations_final)
    opacity_final = pc_opacity_activation(opacity_final)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            assert 0
            # shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            # dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = shs_feature #pc.get_features
    else:
        assert 0,'not merget yet for override_color'
        colors_precomp = override_color
 
    rendered_image, radii, depth,  = rasterizer( 
        colors_precomp = colors_precomp,
        cov3D_precomp = cov3D_precomp,
        # already stack and computed
        means2D = means2D,
        shs = shs,
        # updated based on stacked
        means3D = means3D_final,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        )
    rendered_image_vis = rendered_image.detach().to('cpu')
    
    from utils.scene_utils import vis_torch_img
    vis_img_debug = False
    vis_img_debug = True
    if vis_img_debug:
        if isinstance(single_compo_or_list,list):
            window_topic = f'compo_{single_compo_or_list}_renderObjsOnce'
        else:
            window_topic = f'compo_{single_compo_or_list}_renderObjsSeperately'
        vis_torch_img(rendered_image=rendered_image,
                      topic=window_topic,
                    #   topic = f'compo_{single_compo_or_list}_{single_compo_or_list}',
                    #   save_path = None,
                      )

    # else:
    return {"render": rendered_image,
            "depth": depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,}, single_compo_or_list_idx



