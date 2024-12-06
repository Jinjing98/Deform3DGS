import torch
from utils.sh_utils import eval_sh
from scene.mis_gaussian_model import MisGaussianModel
from utils.camera_utils import Camera

import torch
import math

def make_rasterizer_misgs(
    viewpoint_camera: Camera,
    active_sh_degree = 0,
    bg_color = None,
    scaling_modifier = None,
    cfg = None,
):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    if bg_color is None:
        bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
        bg_color = torch.tensor(bg_color).float().cuda()
    if scaling_modifier is None:
        scaling_modifier = cfg.render.scaling_modifier
    debug = cfg.render.debug
    
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
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=debug,
    )    
            
    rasterizer: GaussianRasterizer = GaussianRasterizer(raster_settings=raster_settings)
    return rasterizer








class MisGaussianRenderer():
    def __init__(
        self,
        cfg = None,         
    ):
        self.cfg_render = cfg.render
        self.cfg = cfg
              
    def render_all(
        self, 
        viewpoint_camera: Camera,
        pc: MisGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):
        
        # render all
        render_composition = self.render(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)

        # render background
        render_background = self.render_background(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        
        # render object
        render_object = self.render_object(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        
        result = render_composition
        result['rgb_background'] = render_background['rgb']
        result['acc_background'] = render_background['acc']
        result['rgb_object'] = render_object['rgb']
        result['acc_object'] = render_object['acc']
        
        # result['bboxes'], result['bboxes_input'] = pc.get_bbox_corner(viewpoint_camera)
    
        return result
    
    def render_object(
        self, 
        viewpoint_camera: Camera,
        pc: MisGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):        
        pc.set_visibility(include_list=pc.obj_list)
        pc.parse_camera(viewpoint_camera)
        
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, white_background=True)

        return result
    
    def render_background(
        self, 
        viewpoint_camera: Camera,
        pc: MisGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):
        pc.set_visibility(include_list=['background_model_name'])
        pc.parse_camera(viewpoint_camera)
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color, white_background=True)

        return result
    
    def render_sky(
        self, 
        viewpoint_camera: Camera,
        pc: MisGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None
    ):  
        pc.set_visibility(include_list=['sky'])
        pc.parse_camera(viewpoint_camera)
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        return result
    
    def render(
        self, 
        viewpoint_camera: Camera,
        pc: MisGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        exclude_list = [],
    ):   
        include_list = list(set(pc.model_name_id.keys()) - set(exclude_list))
                    
        # Step1: render foreground
        pc.set_visibility(include_list)
        pc.parse_camera(viewpoint_camera)
        
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)

        # Step2: render sky
        if pc.include_sky:
            sky_color = pc.sky_cubemap(viewpoint_camera, result['acc'].detach())

            result['rgb'] = result['rgb'] + sky_color * (1 - result['acc'])

        if pc.use_color_correction:
            result['rgb'] = pc.color_correction(viewpoint_camera, result['rgb'])

        if self.cfg.mode != 'train':
            result['rgb'] = torch.clamp(result['rgb'], 0., 1.)

        return result
    
            
    def render_kernel(
        self, 
        viewpoint_camera: Camera,
        pc: MisGaussianModel,
        convert_SHs_python = None, 
        compute_cov3D_python = None, 
        scaling_modifier = None, 
        override_color = None,
        white_background = None,
        # self.cfg.data.white_background,
    ):
        white_background =self.cfg.data.white_background
        
        if pc.num_gaussians == 0:
            if white_background:
                rendered_color = torch.ones(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            else:
                rendered_color = torch.zeros(3, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            
            rendered_acc = torch.zeros(1, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            rendered_semantic = torch.zeros(0, int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), device="cuda")
            
            return {
                "rgb": rendered_color,
                "acc": rendered_acc,
                "semantic": rendered_semantic,
            }

        # Set up rasterization configuration and make rasterizer
        bg_color = [1, 1, 1] if white_background else [0, 0, 0]
        bg_color = torch.tensor(bg_color).float().cuda()
        scaling_modifier = scaling_modifier or self.cfg_render.scaling_modifier
        # from utils.camera_utils import make_rasterizer_misgs
        #///////////////////////////////////////////////////////////////////////////////////////////
        import sys
        # our_sys_path_container = sys.path.copy()
        # assert 0,sys.path
        # sys.path.insert(0, '/mnt/cluster/workspaces/jinjingxu/proj/street_gaussians/lib/utils')
        # sys.path.append('/mnt/cluster/workspaces/jinjingxu/proj/street_gaussians')
        # sys.path.append('/mnt/cluster/workspaces/jinjingxu/proj/street_gaussians/lib/utils/camera_utils')
        # from lib.utils.camera_utils 
        # import make_rasterizer_misgs
        # print('todo ')
        rasterizer = make_rasterizer_misgs(viewpoint_camera, pc.max_sh_degree, bg_color, scaling_modifier,
                                     cfg = self.cfg)
        # sys.path = our_sys_path_container
        #///////////////////////////////////////////////////////////////////////////////////////////
        
        convert_SHs_python = convert_SHs_python or self.cfg_render.convert_SHs_python
        compute_cov3D_python = compute_cov3D_python or self.cfg_render.compute_cov3D_python

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        if self.cfg.mode == 'train':
            screenspace_points = torch.zeros((pc.num_gaussians, 3), requires_grad=True).float().cuda() + 0
            try:
                screenspace_points.retain_grad()
            except:
                pass
        else:
            screenspace_points = None 

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                try:
                    shs = pc.get_features
                except:
                    colors_precomp = pc.get_colors(viewpoint_camera.camera_center)
        else:
            colors_precomp = override_color

        # TODO: add more feature here
        feature_names = []
        feature_dims = []
        features = []
        
        if self.cfg.render.render_normal:
            assert 0
            normals = pc.get_normals(viewpoint_camera)
            feature_names.append('normals')
            feature_dims.append(normals.shape[-1])
            features.append(normals)

        if self.cfg.data.get('use_semantic', False):
            assert 0
            semantics = pc.get_semantic
            feature_names.append('semantic')
            feature_dims.append(semantics.shape[-1])
            features.append(semantics)
        
        if len(features) > 0:
            features = torch.cat(features, dim=-1)
        else:
            features = None
        
        print('******************************************')
        list_to_print = [means2D,means3D,opacity,shs,colors_precomp,scales,rotations,cov3D_precomp]
        for i, ele in enumerate(list_to_print):
            try:
                print(i,ele.dtype,ele.shape,ele.device,ele.requires_grad)
            except:
                print(i,ele)
            # 0 torch.float32 torch.Size([34281, 3]) cuda:0 True [04/12 23:25:09]
            # 1 torch.float32 torch.Size([34281, 3]) cuda:0 True [04/12 23:25:09]
            # 2 torch.float32 torch.Size([34281, 1]) cuda:0 True [04/12 23:25:09]
            # 3 torch.float32 torch.Size([34281, 16, 3]) cuda:0 True [04/12 23:25:09]
            # 4 None [04/12 23:25:09]
            # 5 torch.float32 torch.Size([34281, 3]) cuda:0 True [04/12 23:25:09]
            # 6 torch.float32 torch.Size([34281, 4]) cuda:0 True [04/12 23:25:09]
            # 7 None [04/12 23:25:09]
        print('******************************************')
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        # rendered_color, radii, rendered_depth, rendered_acc, rendered_feature = rasterizer(
        rendered_color, radii, rendered_depth, = rasterizer(
            means3D = means3D,
            means2D = means2D,
            opacities = opacity,
            shs = shs,
            colors_precomp = colors_precomp,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            # semantics = features,
        )  
        rendered_acc =  torch.empty([0]) #torch.empty_like(rendered_depth)
        rendered_feature = torch.Tensor([])#torch.empty([0])
        assert 0,rendered_color
        #////////////////////////////////////////////////////////////////////////////////
        # sys.path = our_sys_path_container



        if self.cfg.mode != 'train':
            rendered_color = torch.clamp(rendered_color, 0., 1.)
        
        rendered_feature_dict = dict()
        if rendered_feature.shape[0] > 0:
            rendered_feature_list = torch.split(rendered_feature, feature_dims, dim=0)
            for i, feature_name in enumerate(feature_names):
                rendered_feature_dict[feature_name] = rendered_feature_list[i]
        
        if 'normals' in rendered_feature_dict:
            rendered_feature_dict['normals'] = torch.nn.functional.normalize(rendered_feature_dict['normals'], dim=0)
                
        if 'semantic' in rendered_feature_dict:
            rendered_semantic = rendered_feature_dict['semantic']
            semantic_mode =self.cfg.model.gaussian.get('semantic_mode', 'logits')
            assert semantic_mode in ['logits', 'probabilities']
            if semantic_mode == 'logits': 
                pass # return raw semantic logits
            else:
                rendered_semantic = rendered_semantic / (torch.sum(rendered_semantic, dim=0, keepdim=True) + 1e-8) # normalize to probabilities
                rendered_semantic = torch.log(rendered_semantic + 1e-8) # change for cross entropy loss

            rendered_feature_dict['semantic'] = rendered_semantic
        
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        
        result = {
            "rgb": rendered_color,
            "acc": rendered_acc,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii
        }
        


        result.update(rendered_feature_dict)
        
        return result