import cv2
import numpy as np
import os

def project_3d_to_2d(pt,K,ob_in_cam):
  pt = pt.reshape(4,1)
  pt_3d = (ob_in_cam@pt)[:3,:]
  projected = K @ pt_3d
  projected = projected.reshape(-1)
  assert projected[2]!=0
  projected = projected/(projected[2])
  return projected.reshape(-1)[:2].round().astype(int)



def draw_xyz_axis(color, ob_in_cam, scale=0.1, K=np.eye(3), thickness=3, transparency=0.3,is_input_rgb=False):
  '''
#   ob_in_cam: pose
  @color: BGR
  '''
  #todo we take the 1st frame as ref--for simplicity, add a delta_trans(3D space) for the ob_in_cam
  if is_input_rgb:
    color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
  xx = np.array([1,0,0,1]).astype(float)
  yy = np.array([0,1,0,1]).astype(float)
  zz = np.array([0,0,1,1]).astype(float)
  xx[:3] = xx[:3]*scale
  yy[:3] = yy[:3]*scale
  zz[:3] = zz[:3]*scale

  origin = tuple(project_3d_to_2d(np.array([0,0,0,1]), K, ob_in_cam))
  
  xx = tuple(project_3d_to_2d(xx, K, ob_in_cam))
  yy = tuple(project_3d_to_2d(yy, K, ob_in_cam))
  zz = tuple(project_3d_to_2d(zz, K, ob_in_cam))
  print( f'the axis are draw with {xx} {yy} {zz} {origin}')
  line_type = cv2.FILLED
  arrow_len = 0
  tmp = color.copy()
  tmp1 = tmp.copy()
  #red
  tmp1 = cv2.arrowedLine(tmp1, origin, xx, color=(0,0,255), thickness=thickness,line_type=line_type, tipLength=arrow_len)
  mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
  tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
  tmp1 = tmp.copy()
  #grenn
  tmp1 = cv2.arrowedLine(tmp1, origin, yy, color=(0,255,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
  mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
  tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
  tmp1 = tmp.copy()
  #blue
  tmp1 = cv2.arrowedLine(tmp1, origin, zz, color=(255,0,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
  mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
  tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
  tmp = tmp.astype(np.uint8)
  if is_input_rgb:
    tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)

  return tmp

if __name__=='__main__':
    # xyz: red /green /blue
    # positive dir: left /up /hit_eye

    # scale=100
    # be default, regard the length of the axis in 3D as 1 meter, then considering the scale
    scale=0.01
    ob_in_cam_dbg = np.eye(4)
    delta_trans_cm = np.zeros_like(ob_in_cam_dbg)
    #0,0,0 if the very center?
    # 0 0 0 mean the obj starts with the lense center location
    # in cm
    delta_trans_cm[:3,-1] = np.array([ 
                                # the abs value: bigger==far away from center in x; posistive is left
                                -3.101, 
                                # the abs value: bigger==far away from center in y; posistive is up
                                1.0,
                                # abs bigger(can't be too small,at least 10), the axis looks smaller (deeper)
                                # can only be negative(positive): its positive and negative will affect location
                                -10, 
                                  ])
    #unit should be meter
    # cm -> m
    delta_trans = delta_trans_cm*0.01
    ob_in_cam_dbg += delta_trans



    ob_in_cam = ob_in_cam_dbg
    
    # 484.6379395           0 283.3963318
    #           0  484.553772 241.3950958
    #           0           0           1

    cx = 283
    cy = 241
    fx = 484
    fy = 484
    K = np.array([
       [fx,0, cx],
       [0, fy, cy],
       [0, 0, 1],
    ])

    # rgb = cv2.resize(rgb, dsize=(self.W,self.H), interpolation=cv2.INTER_LINEAR)
    rgb_img_root = '/mnt/ceph/tco/TCO-All/Projects/MisGS/liwen_rp_output/BundleSDF_output/stereomis/color'
    rgb_img_name = '000246l.png'
    rgb_img_path = os.path.join(rgb_img_root,rgb_img_name)
    
    

    
    assert os.path.exists(rgb_img_path)
    rgb = cv2.imread(rgb_img_path)
    vis = draw_xyz_axis(rgb[...,::-1], ob_in_cam=ob_in_cam, scale=scale, K=K, transparency=0, thickness=5)
    vis = vis[...,::-1]



    # to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    # for color_file in color_files:
    #     color = imageio.imread(color_file)
    #     pose = np.loadtxt(color_file.replace('.png','.txt').replace('color','ob_in_cam'))
    #     pose = pose@np.linalg.inv(to_origin)


    # if self.use_gui:
    #     ob_in_cam = np.linalg.inv(frame._pose_in_model)
 
