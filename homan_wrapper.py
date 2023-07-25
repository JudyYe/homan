import json
from glob import glob
import pickle
import os
import os.path as osp
import numpy as np
from PIL import Image
import trimesh
import numpy as np
from PIL import Image
import torch
from homan.prepare.frameinfos import get_frame_infos
from homan.pointrend import MaskExtractor
from homan.tracking import trackseq
from homan.mocap import get_hand_bbox_detector
from homan.utils.bbox import  make_bbox_square, bbox_xy_to_wh, bbox_wh_to_xy
from libyana.visutils import imagify
from homan.pose_optimization import find_optimal_poses
from homan.lib2d import maskutils
from homan.viz.colabutils import display_video
from homan.jointopt import optimize_hand_object
from jutils import image_utils

import sys
ROOT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, ROOT_DIR + "/external/frankmocap/detectors/hand_object_detector/lib")
sys.path.insert(0, ROOT_DIR + "/external/frankmocap")
from handmocap.hand_mocap_api import HandMocap

# sample_folder = "tmp/"
# step2_folder = os.path.join(sample_folder, "jointoptim_step2")
# step2_viz_folder = os.path.join(step2_folder, "viz")




def main():
  # hoi_wrapper = HomanWrapper('/private/home/yufeiy2/scratch/result/homan/demo',)
  # hoi_wrapper.run()  

  # hoi_wrapper = HomanWrapper("/private/home/yufeiy2/scratch/result/homan/Mug_1", 
  #                            '/private/home/yufeiy2/scratch/result/HOI4D/Mug_1/image', 
  #                            '/private/home/yufeiy2/scratch/result/HOI4D/Mug_1/oObj.obj')
  hoi_wrapper = HomanWrapper("/private/home/yufeiy2/scratch/result/homan/Bottle_1", 
                             '/private/home/yufeiy2/scratch/result/HOI4D/Bottle_1/image', 
                             '/private/home/yufeiy2/scratch/result/HOI4D/Bottle_1/oObj.obj')
  hoi_wrapper.run(True)

class HomanWrapper:
  def __init__(self, 
               sample_folder="tmp", 
               image_folder="images", 
               obj_path="local_data/datasets/shapenetmodels/206ef4c97f50caa4a570c6c691c987a8.obj",
               start_idx=160,
               time=10) -> None:
    self.image_folder = image_folder
    self.obj_path = obj_path
    self.sample_folder = sample_folder
    self.step2_folder = os.path.join(sample_folder, "jointoptim_step2")
    self.step2_viz_folder = os.path.join(self.step2_folder, "viz")

    image_names = sorted(os.listdir(image_folder))
    self.start_idx = start_idx
    if time > 0 and self.start_idx >= len(image_names) - time:
      self.start_idx = len(image_names) - time - 1
    self.time = time if time > 0 else len(image_names)

    os.makedirs(self.sample_folder, exist_ok=True)
  
  def run(self, hijack_gt=False, cfg=None):
    images, images_np, obj_verts_can, obj_faces = self.load_image_cad(self.image_folder, self.obj_path)
    if not hijack_gt:
      hand_bboxes, obj_bboxes = self.preprocess_bbox(images)
      person_parameters, obj_mask_infos, camintrs = self.preprocess_mask(images_np, hand_bboxes, obj_bboxes)
    else:
      hand_bboxes, obj_bboxes = self.load_my_gt_bboxes()    
      person_parameters, obj_mask_infos, camintrs = self.hijack_mask(images_np, hand_bboxes, obj_bboxes)

    object_parameters = self.init_poses(images_np, obj_verts_can, obj_faces, person_parameters, obj_mask_infos, camintrs)

    model =  self.coarse_joint_opt(images_np, person_parameters, object_parameters, obj_verts_can, obj_faces, camintrs, cfg.get('coarse', None))
    model_fine = self.finegrain_joint_opt(images_np, person_parameters, object_parameters, obj_verts_can, obj_faces, camintrs, model, cfg.get('fine', None))

    # save output
    with open(os.path.join(self.sample_folder, "model.pkl"), "wb") as f:
      pickle.dump({
        'hand_bboxes': hand_bboxes,
        'obj_bboxes': obj_bboxes,
        'person_parameters': person_parameters,
        'obj_mask_infos': obj_mask_infos,
        'camintrs': camintrs,
        'object_parameters': object_parameters,
        'model': model,
        'model_fine': model_fine,
        }, f)
    return 
  
  def load_my_gt_bboxes(self, ):
    # dict_keys(['bboxes', 'cams', 'faces', 'local_cams', 'verts', 'verts2d', 'rotations', 'mano_pose', 'mano_pca_pose', 'mano_rot', 'mano_betas', 'mano_trans', 'translations', 'hand_side', 'masks'])
    hand_bbox_list = sorted(glob(os.path.join(self.image_folder, '../hand_boxes', '*.json')))[self.start_idx:self.start_idx + self.time]
    hand_bboxes = [np.array(json.load(open(bbox_path))['hand_bbox_list'][0]['right_hand']).astype(np.float64) for bbox_path in hand_bbox_list]
    sq_hand_bboxes = [make_bbox_square(bbox, 0.1) for bbox in hand_bboxes]
    hand_rtn = {'right_hand': np.array(sq_hand_bboxes)}

    obj_mask_list = sorted(glob(os.path.join(self.image_folder, '../obj_mask', '*.png')))[self.start_idx:self.start_idx + self.time]
    obj_masks = [np.array(Image.open(mask_path)) for mask_path in obj_mask_list]
    obj_bboxes = [image_utils.mask_to_bbox(mask) for mask in obj_masks]
    obj_rtn = np.array(obj_bboxes).astype(np.float64)[None]
    return hand_rtn, obj_rtn

  def simplify_mesh(self, from_path, nsample):
    out_path = osp.join(self.sample_folder, 'simplified.obj')
    cmd = f'/private/home/yufeiy2/Packages/Manifold/build/manifold {from_path} {out_path} {nsample}'
    os.system(cmd)
    return out_path
  
  def load_image_cad(self, image_folder='images', obj_path="local_data/datasets/shapenetmodels/206ef4c97f50caa4a570c6c691c987a8.obj"):
    # Get images in "images" folder according to alphabetical order
    image_names = sorted(os.listdir(image_folder))
    image_paths = [os.path.join(image_folder, image_name) for image_name in image_names[self.start_idx:self.start_idx + self.time]]

    # Convert images to numpy
    images = [Image.open(image_path) for image_path in image_paths if (image_path.lower().endswith(".jpg") or image_path.lower().endswith(".png"))]
    images_np = [np.array(image) for image in images]


    # Visualize the 10 frames
    viz_num_images = 10
    print(f"Loaded {len(images)} images, displaying the first {viz_num_images}")
    imagify.viz_imgrow(images_np[:viz_num_images], path=f'{self.sample_folder}tmp.png')

    # Get local object model
    # Initialize object scale
    obj_scale = 0.08  # Obj dimension in meters (0.1 => 10cm, 0.01 => 1cm)

    obj_path = self.simplify_mesh(obj_path, 2000)

    obj_mesh = trimesh.load(obj_path, force="mesh")
    obj_verts = np.array(obj_mesh.vertices).astype(np.float32)

    # Center and scale vertices
    obj_verts = obj_verts - obj_verts.mean(0)
    obj_verts_can = obj_verts / np.linalg.norm(obj_verts, 2, 1).max() * obj_scale / 2
    obj_faces = np.array(obj_mesh.faces)

    # Display object vertices as scatter plot to visualize object shape
    print(f"Two projections of the centered object vertices, scaled to {obj_scale * 100} cm")
    imagify.viz_pointsrow([obj_verts, obj_verts[:, 1:]], path=f'{self.sample_folder}/tmp.png')

    return images, images_np, obj_verts_can, obj_faces

  def preprocess_bbox(self, images, ):
    # Load object mesh
    hand_detector = get_hand_bbox_detector()
    seq_boxes = trackseq.track_sequence(images, 256, hand_detector=hand_detector, setup={"right_hand": 1, "objects": 1})
    hand_bboxes = {key: make_bbox_square(bbox_xy_to_wh(val), bbox_expansion=0.1) for key, val in seq_boxes.items() if 'hand' in key}
    obj_bboxes = [seq_boxes['objects']]
    return hand_bboxes, obj_bboxes

  def hijack_mask(self, images_np, hand_bboxes, obj_bboxes):
    obj_mask_list = sorted(glob(os.path.join(self.image_folder, '../obj_mask', '*.png')))[self.start_idx:self.start_idx + self.time]
    obj_masks = [np.array(Image.open(mask_path)) for mask_path in obj_mask_list]
    obj_masks = np.stack(obj_masks, 0)
    obj_masks = torch.BoolTensor(obj_masks > 0)
    
    hand_masks = sorted(glob(os.path.join(self.image_folder, '../hand_mask', '*.png')))[self.start_idx:self.start_idx + self.time]
    hand_masks = [np.array(Image.open(mask_path)) for mask_path in hand_masks]
    hand_masks = np.stack(hand_masks, 0)
    hand_masks = hand_masks > 0
    hand_masks = torch.BoolTensor(hand_masks)

    mask_extractor = MaskExtractor(pointrend_model_weights="detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl")
    frankmocap_hand_checkpoint = ROOT_DIR + "/extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
    hand_predictor = HandMocap(frankmocap_hand_checkpoint, ROOT_DIR + "/extra_data/smpl")

    # Define camera parameters
    height, width, _ = images_np[0].shape
    image_size = max(height, width)
    focal = 480
    camintr = np.array([[focal, 0, width // 2], [0, focal, height // 2], [0, 0, 1]]).astype(np.float32)
    camintrs = [camintr for _ in range(len(images_np))]

    # Initialize object motion
    person_parameters, obj_mask_infos, super2d_imgs = get_frame_infos(images_np,
                                                                      mask_extractor=mask_extractor,
                                                                      hand_predictor=hand_predictor,
                                                                      hand_bboxes=hand_bboxes,
                                                                      obj_bboxes=np.stack(obj_bboxes),
                                                                      sample_folder=self.sample_folder,
                                                                      camintr=camintrs,
                                                                      image_size=image_size,
                                                                      debug=False,
                                                                      hand_masks=hand_masks,
                                                                      obj_masks=obj_masks)
    Image.fromarray(super2d_imgs).save(os.path.join(self.sample_folder, "super2d.png"))
    return person_parameters, obj_mask_infos, camintrs

    
  def preprocess_mask(self, images_np, hand_bboxes, obj_bboxes):
    # Initialize segmentation and hand pose estimation models
    mask_extractor = MaskExtractor(pointrend_model_weights="detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl")
    frankmocap_hand_checkpoint = "extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
    hand_predictor = HandMocap(frankmocap_hand_checkpoint, "extra_data/smpl")

    # Define camera parameters
    height, width, _ = images_np[0].shape
    image_size = max(height, width)
    focal = 480
    camintr = np.array([[focal, 0, width // 2], [0, focal, height // 2], [0, 0, 1]]).astype(np.float32)
    camintrs = [camintr for _ in range(len(images_np))]

    # Initialize object motion
    person_parameters, obj_mask_infos, super2d_imgs = get_frame_infos(images_np,
                                                                      mask_extractor=mask_extractor,
                                                                      hand_predictor=hand_predictor,
                                                                      hand_bboxes=hand_bboxes,
                                                                      obj_bboxes=np.stack(obj_bboxes),
                                                                      sample_folder=self.sample_folder,
                                                                      camintr=camintrs,
                                                                      image_size=image_size,
                                                                      debug=False)
    Image.fromarray(super2d_imgs).save(os.path.join(self.sample_folder, "super2d_theirs.png"))
    return person_parameters, obj_mask_infos, camintrs

  def init_poses(self, images_np, obj_verts_can, obj_faces, person_parameters, obj_mask_infos, camintrs):
    object_parameters = find_optimal_poses(
        images=images_np,
        image_size=images_np[0].shape,
        vertices=obj_verts_can,
        faces=obj_faces,
        annotations=obj_mask_infos,
        num_initializations=200,
        num_iterations=10, # Increase to get more accurate initializations
        Ks=np.stack(camintrs),
        viz_path=os.path.join(self.sample_folder, "optimal_pose.png"),
        debug=False,
    )

    # Add object object occlusions to hand masks
    for person_param, obj_param, camintr in zip(person_parameters,
                                            object_parameters,
                                            camintrs):
        maskutils.add_target_hand_occlusions(
            person_param,
            obj_param,
            camintr,
            debug=False,
            sample_folder=self.sample_folder)

    return object_parameters
  
  def coarse_joint_opt(self, images_np, person_parameters, object_parameters, obj_verts_can, obj_faces, camintrs, coarse=None):
    sample_folder = self.sample_folder
    step2_folder = self.step2_folder
    step2_viz_folder = self.step2_viz_folder

    if coarse is not None:
      coarse_num_iterations = coarse.step
      coarse_loss_weights = coarse
    else:      
      coarse_num_iterations = 201 # Increase to give more steps to converge
      coarse_loss_weights = {
              "lw_inter": 1,
              "lw_depth": 1,
              "lw_sil_obj": 1.0,
              "lw_sil_hand": 0.0,
              "lw_collision": 0.0,
              "lw_contact": 0.0,
              "lw_scale_hand": 0.001,
              "lw_scale_obj": 0.001,
              "lw_v2d_hand": 50,
              "lw_smooth_hand": 2000,
              "lw_smooth_obj": 2000,
              "lw_pca": 0.004,
          }

    coarse_viz_step = 10 # Decrease to visualize more optimization steps
    image_size = max(images_np[0].shape[:2])

    # Camera intrinsics in normalized coordinate
    camintr_nc = np.stack(camintrs).copy().astype(np.float32)
    camintr_nc[:, :2] = camintr_nc[:, :2] / image_size

    # Coarse hand-object fitting
    model, loss_evolution, imgs = optimize_hand_object(
        person_parameters=person_parameters,
        object_parameters=object_parameters,
        hand_proj_mode="persp",
        objvertices=obj_verts_can,
        objfaces=np.stack([obj_faces for _ in range(len(images_np))]),
        optimize_mano=True,
        optimize_object_scale=True,
        loss_weights=coarse_loss_weights,
        image_size=image_size,
        num_iterations=coarse_num_iterations + 1,  # Increase to get more accurate initializations
        images=np.stack(images_np),
        camintr=camintr_nc,
        state_dict=None,
        viz_step=coarse_viz_step,
        viz_folder=step2_viz_folder,
    )
    print(f"{coarse_viz_step}, {step2_viz_folder}")
    print(os.path.join(step2_folder, "joint_optim.mp4"))
    print(os.path.join(sample_folder, "jointoptim_step2.mp4"))

    last_viz_idx = (coarse_num_iterations // coarse_viz_step) * coarse_viz_step
    video_step2 = display_video(os.path.join(step2_folder, "joint_optim.mp4"),
                                os.path.join(sample_folder, "jointoptim_step2.mp4"))
    # display(video_step2)
    return model

  def finegrain_joint_opt(self, images_np, person_parameters, object_parameters, obj_verts_can, obj_faces, camintrs, model, fine=None):
    sample_folder = self.sample_folder

    if fine is not None:
      finegrained_num_iterations = fine.step
      finegrained_loss_weights = fine
    else:
      finegrained_num_iterations = 201   # Increase to give more time for convergence
      finegrained_loss_weights = {
              "lw_inter": 1,
              "lw_depth": 1,  # orig: 1
              "lw_sil_obj": 1.0,
              "lw_sil_hand": 0.0,
              "lw_collision": 0.001,
              "lw_contact": 1.0,
              "lw_scale_hand": 0.001,
              "lw_scale_obj": 0.001,
              "lw_v2d_hand": 50,
              "lw_smooth_hand": 2000,
              "lw_smooth_obj": 2000,
              "lw_pca": 0.004,
          }
    finegrained_viz_step = 10 # Decrease to visualize more optimization steps
    image_size = max(images_np[0].shape[:2])

    # Camera intrinsics in normalized coordinate
    camintr_nc = np.stack(camintrs).copy().astype(np.float32)
    camintr_nc[:, :2] = camintr_nc[:, :2] / image_size

    # Refine hand-object fitting
    step3_folder = os.path.join(sample_folder, "jointoptim_step3")
    step3_viz_folder = os.path.join(step3_folder, "viz")
    model_fine, loss_evolution, imgs = optimize_hand_object(
        person_parameters=person_parameters,
        object_parameters=object_parameters,
        hand_proj_mode="persp",
        objvertices=obj_verts_can,
        objfaces=np.stack([obj_faces for _ in range(len(images_np))]),
        optimize_mano=True,
        optimize_object_scale=True,
        loss_weights=finegrained_loss_weights,
        image_size=image_size,
        num_iterations=finegrained_num_iterations + 1,
        images=np.stack(images_np),
        camintr=camintr_nc,
        state_dict=model.state_dict(),
        viz_step=finegrained_viz_step,
        viz_folder=step3_viz_folder,
    )
    last_viz_idx = (finegrained_num_iterations // finegrained_viz_step) * finegrained_viz_step
    video_step3 = display_video(os.path.join(step3_folder, "joint_optim.mp4"),
                                os.path.join(sample_folder, "jointoptim_step3.mp4"))
    return model_fine
  

if __name__ == '__main__':
  main()