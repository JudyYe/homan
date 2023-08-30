from tqdm import tqdm
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
from homan.homan import minimal_cat
from homan.eval import evalviz
from homan.prepare.frameinfos import get_frame_infos
from homan.pointrend import MaskExtractor
from homan.tracking import trackseq
from homan.mocap import get_hand_bbox_detector
from homan.utils.bbox import  make_bbox_square, bbox_xy_to_wh
from libyana.visutils import imagify
from homan.pose_optimization import find_optimal_poses
from homan.utils.geometry import rot6d_to_matrix
from homan.lib2d import maskutils
from homan.visualize import visualize_hand_object
from homan.viz.colabutils import display_video
from homan.jointopt import optimize_hand_object
from jutils import image_utils, mesh_utils, hand_utils, geom_utils
from pytorch3d.structures import Meshes

import sys
ROOT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, ROOT_DIR + "/external/frankmocap/detectors/hand_object_detector/lib")
sys.path.insert(0, ROOT_DIR + "/external/frankmocap")
from handmocap.hand_mocap_api import HandMocap


class VideoChunks:
    def __init__(self, sample_folder, image_folder, obj_path, 
                             start_idx=0, frame_nb=10, skip_step=4,) -> None:
        self.image_folder = image_folder
        self.obj_path = obj_path
        self.sample_folder = sample_folder

        self.image_names = sorted(glob(osp.join(image_folder, '*.*g')))
        self.obj_mask_list = sorted(glob(osp.join(image_folder, '../obj_mask', '*.png')))
        self.hand_mask_list = sorted(glob(osp.join(image_folder, '../hand_mask', '*.png')))
        self.hand_bbox_list = sorted(glob(osp.join(image_folder, '../hand_boxes', '*.json')))
        if osp.exists(osp.join(image_folder, '../cameras_hoi_smooth_100.npz')):
            self.intrinsics_pix = np.load(osp.join(image_folder, '../cameras_hoi_smooth_100.npz'))['K_pix'][..., 0:3, 0:3]
            assert len(self.intrinsics_pix) == len(self.image_names), f"{len(self.intrinsics_pix)}, {len(self.image_names)}"
        else:
            self.intrinsics_pix = None

        self.start_idx = start_idx
        self.frame_nb = frame_nb  # batch_size 
        self.skip_step = skip_step
        self.num_batches = int(np.ceil((len(self.image_names) - self.start_idx) / (self.frame_nb * self.skip_step)))
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        start_idx = self.start_idx + idx * self.frame_nb * self.skip_step
        end_idx = start_idx + self.frame_nb * self.skip_step
        end_idx = min(end_idx, len(self.image_names))

        sample = {}
        sample['image'] = self.image_names[start_idx:end_idx:self.skip_step]
        sample['sample_folder'] = osp.join(self.sample_folder, f'sample{start_idx:04d}')
        if len(self.obj_mask_list) > 0:
            sample['obj_mask'] = self.obj_mask_list[start_idx:end_idx:self.skip_step]
            sample['hand_mask'] = self.hand_mask_list[start_idx:end_idx:self.skip_step]
            sample['hand_bbox'] = self.hand_bbox_list[start_idx:end_idx:self.skip_step]
            sample['camintr'] = self.intrinsics_pix[start_idx:end_idx:self.skip_step]

        return sample
    

class HomanWrapper:
    def __init__(
            self, 
            sample_folder="tmp", 
            image_folder="images", 
            obj_path="local_data/datasets/shapenetmodels/206ef4c97f50caa4a570c6c691c987a8.obj",
            opt_scale=False,
            start_idx=0,
            batch_size=10,
            skip_step=4,) -> None:
        self.image_folder = image_folder
        self.obj_path = obj_path
        self.out_obj_file = osp.join(sample_folder, 'simplified.obj')
        self.opt_scale = opt_scale

        self.parent_folder = sample_folder
        self.sample_folder = sample_folder
        self.dataset = VideoChunks(sample_folder, image_folder, obj_path, start_idx, batch_size, skip_step)

    @property
    def step2_folder(self):
        return osp.join(self.sample_folder, "jointoptim_step2")
    @property
    def step2_viz_folder(self):
        return osp.join(self.step2_folder, "viz")
    
    @staticmethod
    @torch.no_grad()
    def extract_my_hoi(model):
        verts_object, _ = model.get_verts_object()
        faces_object = model.faces_object
        oObj = Meshes(verts_object, faces_object)
        oObj.textures = mesh_utils.pad_texture(oObj, 'white')

        pca = model.mano_pca_pose
        rot = model.mano_rot
        beta = model.mano_betas

        hand_wrapper = hand_utils.ManopthWrapper().to(verts_object.device)
        hA = hand_wrapper.pca_to_pose(pca, add_mean=True)
        
        r,t = hand_utils.cvt_axisang_t_i2o(rot, model.mano_trans)
        inside = geom_utils.axis_angle_t_to_matrix(r, t)
        outside = geom_utils.rt_to_homo(
            rot6d_to_matrix(model.rotations_hand).transpose(-1, -2), 
            model.translations_hand.squeeze(1))
        oTh = outside @ inside
        hTo = geom_utils.inverse_rt(mat=oTh, return_mat=True)
        return hA, beta, hTo, oObj

    def run_video(self, hijack_gt=False, cfg=None):
        prev_model = None
        model_list, image_list = [], []
        for idx in tqdm(range(len(self.dataset)), desc=f"{len(self.dataset)} batches"):
            sample = self.dataset[idx]
            self.sample_folder = sample['sample_folder']
            os.makedirs(self.sample_folder, exist_ok=True)
            model, image_np = self.run(sample, hijack_gt, cfg, prev_model=prev_model)
            prev_model = model
            model_list.append(model)
            image_list.append(np.array(image_np))

        self.sample_folder = self.parent_folder
        images_np = np.concatenate(image_list, 0)
        model_seq = minimal_cat(model_list)
        self.visualize(model_seq, images_np, 'seq')
        with open(os.path.join(self.parent_folder, "model_seq.pkl"), "wb") as f:
            pickle.dump({
                'model': model_seq,
                'images': images_np,
                }, f)
        hA, beta, hTo, oObj = self.extract_my_hoi(model_seq)
        with open(osp.join(self.parent_folder, 'param.pkl'), 'wb') as fp:
            pickle.dump({
                'hA': hA, 'beta': beta, 'hTo': hTo, 'oObj': oObj,
            }, fp)

    def run(self, sample, hijack_gt=False, cfg=None, prev_model=None):
        images, images_np, obj_verts_can, obj_faces = self.load_image_cad(sample, self.obj_path)
        if not hijack_gt:
            hand_bboxes, obj_bboxes = self.preprocess_bbox(images)
            person_parameters, obj_mask_infos, camintrs = self.preprocess_mask(images_np, hand_bboxes, obj_bboxes)
        else:
            hand_bboxes, obj_bboxes = self.load_my_gt_bboxes(sample)    
            person_parameters, obj_mask_infos, camintrs = self.hijack_mask(images_np, sample, hand_bboxes, obj_bboxes)

        if prev_model is not None:
            previous_rotations = rot6d_to_matrix(prev_model.rotations_object[-1:])
            object_parameters = self.init_poses(
                images_np, obj_verts_can, obj_faces, person_parameters, obj_mask_infos, camintrs,
                num_initializations=1, previous_rotations=previous_rotations)
        else:
            object_parameters = self.init_poses(images_np, obj_verts_can, obj_faces, person_parameters, obj_mask_infos, camintrs)

        model =  self.coarse_joint_opt(images_np, person_parameters, object_parameters, obj_verts_can, obj_faces, camintrs, cfg.get('coarse', None))
        self.visualize(model, images_np, 'coarse')
        model_fine = self.finegrain_joint_opt(images_np, person_parameters, object_parameters, obj_verts_can, obj_faces, camintrs, model, cfg.get('fine', None))
        self.visualize(model_fine, images_np, 'fine')

        # save output
        with open(os.path.join(self.sample_folder, "model.pkl"), "wb") as f:
            pickle.dump({
                'hand_bboxes': hand_bboxes,
                'obj_bboxes': obj_bboxes,
                'person_parameters': person_parameters,
                'obj_mask_infos': obj_mask_infos,
                'camintrs': camintrs,
                'object_parameters': object_parameters,
                'images': images_np,
                'model': model,
                'model_fine': model_fine,
                }, f)
        return model_fine, images_np

    @torch.no_grad()    
    def visualize(self, model, images_np, suf):
        image_size = max(images_np[0].shape[:2])
        frontal, top_down = visualize_hand_object(model,
                                                    images_np,
                                                    dist=4,
                                                    viz_len=len(images_np),
                                                    image_size=image_size)
        viz_path = os.path.join(self.sample_folder, f"final_points_{suf}.mp4")        
        clip = np.concatenate([
            np.concatenate([np.stack(images_np), frontal, top_down], 2)
        ], 1)
        evalviz.make_video_np(clip, viz_path, resize_factor=0.5)

    def load_my_gt_bboxes(self, sample):
        hand_bbox_list = sample['hand_bbox']
        hand_bboxes = [np.array(json.load(open(bbox_path))['hand_bbox_list'][0]['right_hand']).astype(np.float64) for bbox_path in hand_bbox_list]
        sq_hand_bboxes = [make_bbox_square(bbox, 0.1) for bbox in hand_bboxes]
        hand_rtn = {'right_hand': np.array(sq_hand_bboxes)}

        obj_mask_list = sample['obj_mask']
        obj_masks = [np.array(Image.open(mask_path)) for mask_path in obj_mask_list]
        obj_bboxes = [image_utils.mask_to_bbox(mask) for mask in obj_masks]
        obj_rtn = np.array(obj_bboxes).astype(np.float64)[None]
        return hand_rtn, obj_rtn

    def simplify_mesh(self, from_path, nsample):
        out_path = self.out_obj_file 
        if osp.exists(out_path):
            return out_path
        cmd = f'/private/home/yufeiy2/Packages/Manifold/build/manifold {from_path} {out_path} {nsample}'
        os.system(cmd)
        return out_path
    
    def load_image_cad(self, sample, obj_path="local_data/datasets/shapenetmodels/206ef4c97f50caa4a570c6c691c987a8.obj"):
        # Get images in "images" folder according to alphabetical order
        image_paths = sample['image']
        images = [Image.open(image_path) for image_path in image_paths if (image_path.lower().endswith(".jpg") or image_path.lower().endswith(".png"))]
        images_np = [np.array(image) for image in images]

        # Visualize the 10 frames
        viz_num_images = min(10, len(images))
        print(f"Loaded {len(images)} images, displaying the first {viz_num_images}")
        imagify.viz_imgrow(images_np[:viz_num_images], path=f'{self.sample_folder}/input.png')

        # Get local object model
        # Initialize object scale
        obj_path = self.simplify_mesh(obj_path, 2000)
        obj_mesh = trimesh.load(obj_path, force="mesh")
        obj_verts = np.array(obj_mesh.vertices).astype(np.float32)

        # Center and scale vertices
        obj_verts = obj_verts - obj_verts.mean(0)
        if self.opt_scale:
            obj_scale = 0.08  # Obj dimension in meters (0.1 => 10cm, 0.01 => 1cm)
            obj_verts_can = obj_verts / np.linalg.norm(obj_verts, 2, 1).max() * obj_scale / 2
            print(f"Two projections of the centered object vertices, scaled to {obj_scale * 100} cm")
        else:
            obj_verts_can = obj_verts
        obj_faces = np.array(obj_mesh.faces)

        # Display object vertices as scatter plot to visualize object shape
        imagify.viz_pointsrow([obj_verts, obj_verts[:, 1:]], path=f'{self.sample_folder}/tmp.png')

        return images, images_np, obj_verts_can, obj_faces

    def preprocess_bbox(self, images, ):
        # Load object mesh
        hand_detector = get_hand_bbox_detector()
        seq_boxes = trackseq.track_sequence(images, 256, hand_detector=hand_detector, setup={"right_hand": 1, "objects": 1})
        hand_bboxes = {key: make_bbox_square(bbox_xy_to_wh(val), bbox_expansion=0.1) for key, val in seq_boxes.items() if 'hand' in key}
        obj_bboxes = [seq_boxes['objects']]
        return hand_bboxes, obj_bboxes

    def hijack_mask(self, images_np, sample, hand_bboxes, obj_bboxes):
        obj_mask_list = sample['obj_mask']
        obj_masks = [np.array(Image.open(mask_path)) for mask_path in obj_mask_list]
        obj_masks = np.stack(obj_masks, 0)
        obj_masks = torch.BoolTensor(obj_masks > 0)
        
        hand_masks = sample['hand_mask']
        hand_masks = [np.array(Image.open(mask_path)) for mask_path in hand_masks]
        hand_masks = np.stack(hand_masks, 0)
        hand_masks = hand_masks > 0
        hand_masks = torch.BoolTensor(hand_masks)

        mask_extractor = None # MaskExtractor(pointrend_model_weights="detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl")
        frankmocap_hand_checkpoint = ROOT_DIR + "/extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
        hand_predictor = HandMocap(frankmocap_hand_checkpoint, ROOT_DIR + "/extra_data/smpl")

        # Define camera parameters
        camintrs = sample['camintr']
        height, width, _ = images_np[0].shape
        image_size = max(height, width)

        # Initialize object motion
        person_parameters, obj_mask_infos, super2d_imgs = get_frame_infos(
            images_np,
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
        person_parameters, obj_mask_infos, super2d_imgs = get_frame_infos(
            images_np,
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

    def init_poses(self, images_np, obj_verts_can, obj_faces, 
                   person_parameters, obj_mask_infos, camintrs, 
                   num_initializations=200, previous_rotations=None):
        object_parameters = find_optimal_poses(
                images=images_np,
                image_size=images_np[0].shape,
                vertices=obj_verts_can,
                faces=obj_faces,
                annotations=obj_mask_infos,
                num_initializations=num_initializations,
                num_iterations=10, # Increase to get more accurate initializations
                Ks=np.stack(camintrs),
                viz_path=os.path.join(self.sample_folder, "optimal_pose.png"),
                debug=False,
                previous_rotations=previous_rotations,
        )

        # Add object object occlusions to hand masks
        for person_param, obj_param, camintr in zip(person_parameters, object_parameters, camintrs):
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
                optimize_object_scale=self.opt_scale,
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
                optimize_object_scale=self.opt_scale,
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
    
