ModelParams = dict(
    extra_mark = 'endonerf',
    camera_extent = 10,
    #to-do:
    #    supprot bash edit
    #    support debug dst opt dur
    #    support run for muti splits
    #    support hard changes iters
    init_mode = 'MAPF',#'MAPF', #'skipMAPF' #rand
    tool_mask = 'use', #'use'(default) 'inverse' 'nouse'
)


OptimizationParams = dict(
    coarse_iterations = 0,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    iterations = 1030,
    percent_dense = 0.01,
    opacity_reset_interval = 3000,
    position_lr_max_steps = 4000,
    prune_interval = 3000, #? wrong-not-used?

    #jj
    densification_interval = 100,
    densify_from_iter = 500,
    densify_until_iter = 15_000,
    densify_grad_threshold_coarse = 0.0002,
    densify_grad_threshold_fine_init = 0.0002,
    densify_grad_threshold_after = 0.0002,
    pruning_from_iter = 500,
    pruning_interval = 100,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,  

)

ModelHiddenParams = dict(
    # set to 0 to disable FDM? render also need to set this to 0
    curve_num = 17,#17, # number of learnable basis functions. This number was set to 17 for all the experiments in paper (https://arxiv.org/abs/2405.17835)

    ch_num = 10, # channel number of deformable attributes: 10 = 3 (scale) + 3 (mean) + 4 (rotation)
    init_param = 0.01, )


# PipelineParams = dict(
#     save_iterations = [0,3000], 
    
#     )


# class PipelineParams(ParamGroup):
#     def __init__(self, parser):
#         self.convert_SHs_python = False
#         self.compute_cov3D_python = False
#         self.debug = False
#         super().__init__(parser, "Pipeline Parameters")
