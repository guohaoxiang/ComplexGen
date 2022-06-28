import os
from pickle import decode_long
from random import sample
import sys
import argparse
import time
from datetime import datetime
import math
from tkinter import E
import numpy
from chamferdist import knn_points
from chamferdist import knn_gather
from chamferdist import list_to_padded
from numpy.random.mtrand import f
import trimesh

from numpy.core.defchararray import _join_dispatcher
from torch.utils.tensorboard import SummaryWriter


if(os.path.exists("/blob")):
  #for Philly training
  running_onCluster = True
  print("detected /blob directory, execute on cluster!")
else:
  running_onCluster = False
  print("not detected /blob directory, execute locally")
  from tqdm import tqdm
  from load_ply import *

m = 32 #Unet number of features
#HyperNets parameters
hn_pe_dim = 64
hn_mlp_dim = 64
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    
    parser.add_argument('--no_output', action='store_true', help = 'not output for evaluation')
    parser.add_argument('--reuseid', action='store_true', help = 'reuse id for distance computation')
    parser.add_argument('--ori_topo', action = 'store_true', help = 'original topo embed')
    parser.add_argument('--ori_mlp', action='store_true', help = 'use original version of MLPs')
    parser.add_argument('--no_tri', action='store_true', help = 'not using tripath (not used 20211014)')
    parser.add_argument('--ckpt_interval', default=3000, type=int)
    # parser.add_argument('--dist_th', default=0.05, type=float)
    parser.add_argument('--dist_th', default=0.1, type=float) #modified 1217
    parser.add_argument('--dist_th_tg', default=0.1, type=float) #modified 0107, topo geom consistency
    parser.add_argument('--val_th', default=0.5, type=float)
    # parser.add_argument('--val_th', default=0.3, type=float)
    parser.add_argument('--flag_cycleid', action = 'store_true', help = 'cycle id')
    parser.add_argument('--parsenet', action = 'store_true', help = 'use parsenet data')
    parser.add_argument('--decouple_pos_content', action = 'store_true', help = 'use parsenet data')

    parser.add_argument('--ourresnet', action = 'store_true', help = 'use parsenet data')
    parser.add_argument('--backbone_bn', action = 'store_true', help = 'use parsenet data')
    parser.add_argument('--decoder_rec', action = 'store_true', help = 'use parsenet data')
    parser.add_argument('--m', default=64, type=int, help = 'set m value')
    parser.add_argument('--hidden_dim_mlp', default=384, type=int, help = 'hidden dimension of MLP for ablation study')
  
    #for hn
    parser.add_argument('--hn_scale', action = 'store_true', help = 'original topo embed')
    parser.add_argument('--ori_tripath', action = 'store_true', help = 'original topo embed')

    parser.add_argument('--no_tripath', action = 'store_true', help = 'no tripath, for ablation')
    parser.add_argument('--no_topo', action = 'store_true', help = 'no topo, for ablation, please also set no_tripath as true')


    parser.add_argument('--pe_sin', action = 'store_true', help = 'sin positional embedding')
    parser.add_argument('--pe_sin_base', default=1.2, type=float)
    parser.add_argument('--no_pe', action = 'store_true', help = 'not using positional encoding')
    parser.add_argument('--spe', action = 'store_true', help = 'simple positional encoding')
    # parser.add_argument('--patch_normal', default=1.2, type=float)
    parser.add_argument('--patch_normal', action = 'store_true', help = 'add tangent normal constraints for patch')
    parser.add_argument('--patch_lap', action = 'store_true', help = 'add laplacian constraints for patch')
    parser.add_argument('--patch_lapboundary', action = 'store_true', help = 'add boundary laplacian constraints for patch')
    parser.add_argument('--data_medium', action = 'store_true', help = 'add boundary laplacian constraints for patch')
    parser.add_argument('--vis_train', action = 'store_true', help = 'visualize training data')
    parser.add_argument('--vis_test', action = 'store_true', help = 'visualize test data')
    parser.add_argument('--eval_train', action = 'store_true', help = 'evaluate training data')
    parser.add_argument('--geom_l2', action = 'store_true', help = 'use l2 norm for geometric terms')
    parser.add_argument('--patch_grid', action = 'store_true', help = 'using patch grid')
    parser.add_argument('--patch_close', action = 'store_true', help = 'predict patch closeness')
    parser.add_argument('--batch_cd', action = 'store_true', help = 'compute chamfer distance in batch')
    parser.add_argument('--patch_emd', action = 'store_true', help = 'using emd for patch loss computing, not implemented yet')
    parser.add_argument('--patch_uv', action = 'store_true', help = 'compute patch uv, and patch emd is computed based on patch uv')
    parser.add_argument('--curve_open_loss', action = 'store_true', help = 'using emd for patch loss computing, not implemented yet')
    parser.add_argument('--backbone_expand', action = 'store_true', help = 'expand backbone coordinates and kernel size of the first convolution')

    parser.add_argument('--output_normal', action = 'store_true', help = 'output normal for prediction')
    parser.add_argument('--output_normal_diff_coef', default=1, type=float, help="loss coefficient for output normal diff loss")
    parser.add_argument('--output_normal_tangent_coef', default=1, type=float, help="loss coefficient for output normal tangent lonss")
    parser.add_argument('--enable_automatic_restore', action='store_true', help = 'find ckpt automatically when training is interrupted')
    parser.add_argument('--quicktest', action='store_true', help = 'only test on 10 models, no validation is used')
    # parser.add_argument('--noise', action='store_true', help = 'add noise')
    parser.add_argument('--noise', default=0, type=int, help = 'add noise, 0:no, 1: 0.01, 2: 0.02, 3: 0.05')
    parser.add_argument('--noisetest', default=0, type=int, help = 'add noise for testing, 0:no, 1: 0.01, 2: 0.02, 3: 0.05')
    parser.add_argument('--partial', action='store_true', help = 'use partial data')
    parser.add_argument('--experiment_name', type=str, required = True)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=5000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--points_per_patch_dim', default=20, type=int)
    parser.add_argument('--eval_res_cov', action='store_true', help="evaluate residual loss and coverage")
    parser.add_argument('--eval_matched', action='store_true', help="evaluate residual loss and coverage", default=True)
    parser.add_argument('--eval_selftopo', action='store_true', help="evaluate self topo consistency")
    parser.add_argument('--th_res', default=0.05, type=float, help="threshold for evaluating residual")
    parser.add_argument('--eval_param', action='store_true', help="evaluate residual and converage by parameters")
    parser.add_argument('--evalrest', action='store_true', help="evaluate rest data of 900 models")
    parser.add_argument('--part', default=-1, type=int) #0,1,2,3, divide data into 4 groups
    parser.add_argument('--regen', action='store_true', help="regen files")
    parser.add_argument('--wsa', action='store_true', help="working on wsa")
    parser.add_argument('--th_cov', default=0.01, type=float)
    parser.add_argument('--rotation_augment', action='store_true', help="enable rotation augmentation")
    parser.add_argument('--num_angles', type=int)
    parser.add_argument('--random_angle', action='store_true', help="enable rotation augmentation with random angle")
    parser.add_argument('--input_voxel_dim', default=128, type=int, help="voxel dimension of input")
    
    #if this is the run for evaluation
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--evalfinal', action='store_true')
    parser.add_argument('--evaltopo', action='store_true')
    parser.add_argument('--fittingonce', action='store_true')
    
    # parser.add_argument('--hidden_dim', default=m*6, type=int,
    #                     help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    #parser.add_argument('--num_queries', default=100, type=int,help="Number of query slots")

    parser.add_argument('--num_corner_queries', default=100, type=int,help="Number of query slots")

    parser.add_argument('--num_curve_queries', default=150, type=int,help="Number of query slots")

    parser.add_argument('--num_patch_queries', default=100, type=int,help="Number of query slots")

    parser.add_argument('--pre_norm', action='store_false') #true
    
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    
    # * BackBone unused
    parser.add_argument('--backbone_feature_encode', action='store_true',
                        help="Using sin to encode features in backbone")
    
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--local_attention', dest='using_local_attention',action='store_true',
                        help="Using local attention in transformer")
    
    #correspondences across different level primitives
    parser.add_argument("--topo_embed_dim", default=256, type=int, help="Feature Dimension Size For Topology Matching")
    parser.add_argument("--normalize_embed_feature", action="store_true", help="Normalize Topo Feature before Matching")
    parser.add_argument("--num_heads_dot", default=1, type=int, help="number of heads to compute similarity")
    parser.add_argument("--matrix_eigen_similarity", action="store_true", help="Using Matrix Eigen Similarity")
    # * Loss coefficients
    parser.add_argument('--class_loss_coef', default=1, type=float)
    parser.add_argument('--corner_geometry_loss_coef', default=1000, type=float, help="loss coefficient for geometric loss in corner matching and training")
    parser.add_argument('--curve_geometry_loss_coef', default=1000, type=float, help="loss coefficient for geometric loss in curve matching and training")
    parser.add_argument('--patch_geometry_loss_coef', default=1000, type=float, help="loss coefficient for geometric loss in patch matching and training")
    parser.add_argument('--corner_avg_count', default=20.25, type=float, help="avg corner count for parsenet dataset")
    parser.add_argument('--curve_avg_count', default=37.39, type=float, help="avg curve count for parsenet dataset")
    parser.add_argument('--patch_avg_count', default=18.17, type=float, help="avg patch count for parsenet dataset")
    parser.add_argument('--global_invalid_weight', default=1.0, type=float, help="avg patch count for parsenet dataset")

    
    parser.add_argument('--curve_corner_topo_loss_coef', default=1, type=float)
    parser.add_argument('--patch_curve_topo_loss_coef', default=1, type=float)
    parser.add_argument('--patch_corner_topo_loss_coef', default=1, type=float)
    parser.add_argument('--topo_loss_coef', default=1, type=float)
    parser.add_argument('--curve_corner_geom_loss_coef', default=0, type=float)
    parser.add_argument('--topo_acc', action='store_true',help="compute and show topo_acc")
    parser.add_argument('--no_show_topo', action='store_true',help="not show three topo loss: curve_point, curve_patch, patch_close")

    parser.add_argument('--patch_normal_loss_coef', default=1, type=float, help="loss coefficient for patch normal loss")

    parser.add_argument('--patch_lap_loss_coef', default=1000, type=float, help="loss coefficient for patch normal loss")


    parser.add_argument('--weight_decay', default=1e-4, type=float)
    
    #transformer feature embedding
    parser.add_argument("--curve_embedding_mlp_layers", default=3, type=int)
    
    # Loss
    # parser.add_argument('--no_aux_loss', dest='enable_aux_loss', action='store_false',
    #                     help="Disables auxiliary decoding losses (loss at each layer)") #only enable_aux_loss is callable

    parser.add_argument('--enable_aux_loss', action='store_true',
                        help="Disables auxiliary decoding losses (loss at each layer)") #only enable_aux_loss is callable
    
    # training
    parser.add_argument('--gpu', default="0,1,2", type=str,
                        help="gpu id to be used")
    
    parser.add_argument("--checkpoint_path", default=None, type=str, help="checkpoint file (if have) to be used")
    
    parser.add_argument("--input_feature_type", default='global', type=str, help="input feature type(supported type: local global occupancy)")
    
    parser.add_argument("--input_normal_signals", action='store_true', help='input normal signals in voxel features')
    
    parser.add_argument('--max_training_iterations', default=250001, type=int)
    
    #
    parser.add_argument('--skip_transformer_encoder', action='store_false', help = 'remove encoder part of transformer')
    parser.add_argument('--clip_max_norm', default=0.0, type=float,
                       help='gradient clipping max norm')    

    parser.add_argument('--clip_value', action='store_true', help = 'clip value')
    
    
    parser.add_argument('--single_dir_patch_chamfer', action='store_true', help = 'Single direction chamfer loss in patch processing')

    parser.add_argument('--extra_single_chamfer', action='store_true', help = 'based on emd, add extra single chamfer distance from gt patch to predicted grid')

    parser.add_argument('--extra_single_chamfer_weight', default=300.0, type=float)


    # * Matcher
    parser.add_argument("--using_prob_in_matching", action='store_true', help = 'using -p in matching cost')

    parser.add_argument("--save_gt", action='store_true', help = 'save gt info in predicted pickle file')
    
    parser.add_argument("--no_instance_norm", action='store_true', help = 'using instance normalization in mink backbone')
    
    parser.add_argument("--sin", action='store_true', help = 'using sin activation in geometry mlp')

    parser.add_argument("--suffix", default='_opt_mix_final.json', type=str, help="suffix for evaluation")

    parser.add_argument("--folder", default=None, type=str, help="inter folder for evaluation")

    parser.add_argument("--vis_inter_layer", default=-1, type=int)


    '''
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    '''
    return parser


parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
print("enable_aux_loss:{}".format(args.enable_aux_loss))
m = args.m
print('m value: ', m)

print('eval matches:', args.eval_matched)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
num_of_gpus = len(args.gpu.split(","))
print("Utilize {} gpus".format(num_of_gpus))

from data_loader_abc import *
import MinkowskiEngine as ME
if not args.no_instance_norm: #false
  print("using instance norm")
  import mink_resnet_in as resnets
else:
  import mink_resnet as resnets
import torch.nn as nn
import math
import torch.nn.functional as F

from transformer3d import build_transformer, build_transformer_tripath, build_transformer_debug
from matcher_corner import build_matcher_corner
from matcher_curve import build_matcher_curve, cyclic_curve_points
from matcher_patch import build_matcher_patch, emd_by_id

import torch.multiprocessing as mp

voxel_dim = args.input_voxel_dim
out_voxel_dim = voxel_dim // 8 #16

import torch.distributed as dist
import tensorflow as tf

import plywrite

if args.eval_param:
  from src.primitives import ComputePrimitiveDistance

points_per_curve = 34
points_per_patch_dim = args.points_per_patch_dim #10*10 points per patch

corner_eos_coef_cal = args.corner_avg_count / (args.num_corner_queries - args.corner_avg_count) * args.global_invalid_weight
curve_eos_coef_cal = args.curve_avg_count / (args.num_curve_queries - args.curve_avg_count) * args.global_invalid_weight
patch_eos_coef_cal = args.patch_avg_count / (args.num_patch_queries - args.patch_avg_count) * args.global_invalid_weight

# print('!!!!!!!corner eos: {} curve eos: {} patch eos: {}'.format(corner_eos_coef_cal, curve_eos_coef_cal, patch_eos_coef_cal))

curve_type_list = np.array(['Circle', 'BSpline', 'Line', 'Ellipse'])
patch_type_list = np.array(['Cylinder', 'Torus', 'BSpline', 'Plane', 'Cone', 'Sphere'])

curve_colormap = {'Circle': np.array([255,0,0]), 'BSpline': np.array([255,255,0]), 'Line': np.array([0,255,0]), 'Ellipse': np.array([0,0,255])}

perform_profile = False
profile_iter = 100
profile_dict = {'data_preparation':[], 'network_forwarding':[], 'backbone_forwarding':[], 'transformer_forwarding':[], 'embedding_forwarding':[], 'loss_computation':[], 'gradient_computation':[]}

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_patch_mesh_faces(dimu, dimv, uclose, begin_id = 0):
  faces = []
  for i in range(dimu - 1):
    for j in range(dimv - 1):
      id0 = i * dimv + j + begin_id
      id1 = (i + 1) * dimv + j + begin_id
      id2 = (i + 1) * dimv + j + 1 + begin_id
      id3 = i * dimv + j + 1 + begin_id
      faces.append([id0, id1, id2])
      faces.append([id0, id2, id3])
  
  if uclose:
    for j in range(dimv - 1):
      id0 = (dimu - 1) * dimv + j + begin_id
      id1 =  j + begin_id
      id2 =  j + 1 + begin_id
      id3 = (dimu - 1) * dimv + j + 1 + begin_id
      faces.append([id0, id1, id2])
      faces.append([id0, id2, id3])
  return np.array(faces)

# def normalize_numpy_pt(pt):
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def get_patch_mesh_pts_faces(pts, dimu, dimv, uclose, begin_id = 0, flag_extend = False, extend_offset = 0.05):
  if not flag_extend:
    pts_new = pts
  else:
    pts_new = []
    #update u v
    # if uclose:
    pts = pts.reshape(dimu, dimv, 3)
    uneg_dir = pts[:,1] - pts[:,0]
    uneg_dir = normalized(uneg_dir)
    # print('uneg dir: ', uneg_dir)
    uneg = pts[:, 0] - uneg_dir * extend_offset
    upos_dir = pts[:,-1] - pts[:,-2]
    upos_dir = normalized(upos_dir)
    upos = pts[:,-1] + upos_dir * extend_offset
    
    if uclose:
      uneg = uneg.reshape(dimu, 1, 3)
      upos = upos.reshape(dimu, 1, 3)
      pts_new = np.concatenate([uneg, pts, upos], axis = 1)
      dimv += 2
      pts_new = pts_new.reshape(-1, 3)
    else:
      vneg_dir = pts[1] - pts[0]
      vneg_dir = normalized(vneg_dir)
      vneg = pts[0] - vneg_dir * extend_offset
      vpos_dir = pts[-1] - pts[-2]
      vpos_dir = normalized(vpos_dir)
      vpos = pts[-1] + vpos_dir * extend_offset
      unegneg_dir = uneg[1] - uneg[0]
      unegneg_dir = normalized(unegneg_dir)
      unegneg = uneg[0] - extend_offset * unegneg_dir
      
      unegpos_dir = uneg[-1] - uneg[-2]
      unegpos_dir = normalized(unegpos_dir)
      unegpos = uneg[-1] + extend_offset * unegpos_dir

      uposneg_dir = upos[1] - upos[0]
      uposneg_dir = normalized(uposneg_dir)
      uposneg = upos[0] - extend_offset * uposneg_dir
      upospos_dir = upos[-1] - upos[-2]
      upospos_dir = normalized(upospos_dir)
      upospos = upos[-1] + extend_offset * upospos_dir
      
      uneg = uneg.reshape(dimu, 1, 3)
      upos = upos.reshape(dimu, 1, 3)
      middle = np.concatenate([uneg, pts, upos], axis = 1)
      # dimv += 2
      # pts_new = pts_new.reshape(-1, 3)
      vneg = np.concatenate([unegneg.reshape(1,-1), vneg, uposneg.reshape(1,-1)]).reshape(1, -1, 3)
      vpos = np.concatenate([unegpos.reshape(1,-1), vpos, upospos.reshape(1,-1)]).reshape(1,-1,3)
      pts_new = np.concatenate([vneg, middle, vpos], axis = 0)

      dimv += 2
      dimu += 2
      pts_new = pts_new.reshape(-1,3)

  faces = []
  for i in range(dimu - 1):
    for j in range(dimv - 1):
      id0 = i * dimv + j + begin_id
      id1 = (i + 1) * dimv + j + begin_id
      id2 = (i + 1) * dimv + j + 1 + begin_id
      id3 = i * dimv + j + 1 + begin_id
      faces.append([id0, id1, id2])
      faces.append([id0, id2, id3])
  
  if uclose:
    for j in range(dimv - 1):
      id0 = (dimu - 1) * dimv + j + begin_id
      id1 =  j + begin_id
      id2 =  j + 1 + begin_id
      id3 = (dimu - 1) * dimv + j + 1 + begin_id
      faces.append([id0, id1, id2])
      faces.append([id0, id2, id3])
  return pts_new, np.array(faces)

def compute_overall_singlecd(patch_grids, patch_uclosed, input_pointcloud, matching_indices):
  vert_id = 0
  all_faces = []
  all_pts = []
  # print('effective psize: ', effective_patch_points.shape)
  # print('patch close: ', patch_uclosed)
  assert(patch_grids.shape[0] == patch_uclosed.shape[0])
  # print('grid shape: ', patch_grids.shape[0]) #slightly for topo results and prediction results, because of the threshold setting.
  
  #replace to filtered id since 0122
  set_match_id = set(matching_indices['indices'][0][0].tolist())
  #pred version
  all_pts_matched = []
  all_faces_matched = []
  vert_id_matched = 0
  for i in range(len(patch_grids)):
    # faces = get_patch_mesh_faces(points_per_patch_dim, points_per_patch_dim, patch_uclosed[i], vert_id)
    # faces = get_patch_mesh_faces(points_per_patch_dim, points_per_patch_dim, patch_uclosed[i], vert_id)
    pts, faces = get_patch_mesh_pts_faces(patch_grids[i], points_per_patch_dim, points_per_patch_dim, patch_uclosed[i], vert_id, True, 0.05)
    all_faces.append(faces)
    # vert_id += points_per_patch_dim * points_per_patch_dim
    vert_id += pts.shape[0]
    all_pts.append(pts)
    if args.eval_matched and i in set_match_id:
      pts, faces = get_patch_mesh_pts_faces(patch_grids[i], points_per_patch_dim, points_per_patch_dim, patch_uclosed[i], vert_id_matched, True, 0.05)
      all_faces_matched.append(faces)
      # vert_id += points_per_patch_dim * points_per_patch_dim
      vert_id_matched += pts.shape[0]
      all_pts_matched.append(pts)
  
  # all_faces = np.array(all_faces).reshape(-1, 3)
  all_faces = np.concatenate(all_faces)
  all_pts = np.concatenate(all_pts)
  
  mesh = trimesh.Trimesh(vertices = all_pts, faces = all_faces)
  # mesh.export('allpatch.obj')

  (closest_points,distances,triangle_id) = mesh.nearest.on_surface(input_pointcloud[:,:3])
  #return single_cd, pcov001, pcov002
  if not args.eval_matched:
    return distances
  else:
    all_faces_matched = np.concatenate(all_faces_matched)
    all_pts_matched = np.concatenate(all_pts_matched)
    mesh_matched = trimesh.Trimesh(vertices = all_pts_matched, faces = all_faces_matched)
    (closest_points,distances_matched,triangle_id) = mesh_matched.nearest.on_surface(input_pointcloud[:,:3])

    return distances, distances_matched

def compute_overall_singlecd(patch_grids, patch_uclosed, input_pointcloud, patch_idx_filter):
  vert_id = 0
  all_faces = []
  all_pts = []
  # print('effective psize: ', effective_patch_points.shape)
  # print('patch close: ', patch_uclosed)
  assert(patch_grids.shape[0] == patch_uclosed.shape[0])
  # print('grid shape: ', patch_grids.shape[0]) #slightly for topo results and prediction results, because of the threshold setting.
  
  #replace to filtered id since 0122
  set_match_id = patch_idx_filter
  #pred version
  all_pts_matched = []
  all_faces_matched = []
  vert_id_matched = 0
  for i in range(len(patch_grids)):
    # faces = get_patch_mesh_faces(points_per_patch_dim, points_per_patch_dim, patch_uclosed[i], vert_id)
    # faces = get_patch_mesh_faces(points_per_patch_dim, points_per_patch_dim, patch_uclosed[i], vert_id)
    pts, faces = get_patch_mesh_pts_faces(patch_grids[i], points_per_patch_dim, points_per_patch_dim, patch_uclosed[i], vert_id, True, 0.05)
    all_faces.append(faces)
    # vert_id += points_per_patch_dim * points_per_patch_dim
    vert_id += pts.shape[0]
    all_pts.append(pts)
    if args.eval_matched and i in set_match_id:
      pts, faces = get_patch_mesh_pts_faces(patch_grids[i], points_per_patch_dim, points_per_patch_dim, patch_uclosed[i], vert_id_matched, True, 0.05)
      all_faces_matched.append(faces)
      # vert_id += points_per_patch_dim * points_per_patch_dim
      vert_id_matched += pts.shape[0]
      all_pts_matched.append(pts)
  
  # all_faces = np.array(all_faces).reshape(-1, 3)
  all_faces = np.concatenate(all_faces)
  all_pts = np.concatenate(all_pts)
  
  mesh = trimesh.Trimesh(vertices = all_pts, faces = all_faces)
  # mesh.export('allpatch.obj')

  (closest_points,distances,triangle_id) = mesh.nearest.on_surface(input_pointcloud[:,:3])
  #return single_cd, pcov001, pcov002
  if not args.eval_matched:
    return distances
  else:
    if len(all_faces_matched) == 0:
        return distances, distances

    all_faces_matched = np.concatenate(all_faces_matched)
    all_pts_matched = np.concatenate(all_pts_matched)
    mesh_matched = trimesh.Trimesh(vertices = all_pts_matched, faces = all_faces_matched)
    (closest_points,distances_matched,triangle_id) = mesh_matched.nearest.on_surface(input_pointcloud[:,:3])

    return distances, distances_matched



def compute_overall_singlecd_param(pred_data, input_pointcloud, matching_indices):
  #input pointcloud is a numpy
  #firstly, the grid parts
  patch_close_logits = pred_data['closed_patch_logits'][0].detach().cpu().numpy()
  patch_uclosed = patch_close_logits[:,0] < patch_close_logits[:,1]
  
  if args.eval_matched:
    distances, distances_matched = compute_overall_singlecd(pred_data['pred_patch_points'][0].detach().cpu().numpy(), patch_uclosed, input_pointcloud, matching_indices)
  else:
    distances = compute_overall_singlecd(pred_data['pred_patch_points'][0].detach().cpu().numpy(), patch_uclosed, input_pointcloud, matching_indices)

  # print('distance shape: ', distances.shape)
  #param part
  cp_distance = ComputePrimitiveDistance(reduce = False)
  routines = {
            5: cp_distance.distance_from_sphere,
            0: cp_distance.distance_from_cylinder,
            4: cp_distance.distance_from_cone,
            3: cp_distance.distance_from_plane,
          }

  src_patch_points = pred_data['pred_patch_points'][0]
  src_with_param = pred_data['pred_patch_with_param'][0]
  src_type_logits = pred_data['pred_patch_type'][0]
  src_param = pred_data['pred_patch_param'][0]
  input_pointcloud_torch = torch.tensor(input_pointcloud, device = src_patch_points.device)
  # print('input pc shape: ',input_pointcloud_torch.shape)
  all_dists = []
  all_dists_matched = []
  set_match_id = set(matching_indices['indices'][0][0].tolist())
  print('all patch size: {} matched size: {}'.format(len(src_patch_points), len(set_match_id)))
  for patch_idx in range(len(src_patch_points)):
    if args.eval_param and src_with_param[patch_idx] > 0.5:
      para_dist = routines[torch.argmax(src_type_logits[patch_idx]).item()](input_pointcloud_torch[:,:3], src_param[patch_idx], sqrt = True)
      all_dists.append(para_dist.view([1, -1]))
      if args.eval_matched and patch_idx in set_match_id:
        all_dists_matched.append(para_dist.view([1, -1]))
  
  res = distances
  if len(all_dists) > 0:
    all_dists = torch.cat(all_dists, axis = 0)
    # print('all_dists_shape: ', all_dists.shape)
    distances_all = np.concatenate([distances.reshape(1,-1), all_dists.detach().cpu().numpy()], axis = 0)

    # return distances_all.min(0)
    res = distances_all.min(0)
  
  if args.eval_matched:
    res_matched = distances_matched
    if len(all_dists_matched) > 0:
      all_dists_matched = torch.cat(all_dists_matched, axis = 0)
      # print('all_dists_shape: ', all_dists.shape)
      distances_all = np.concatenate([distances_matched.reshape(1,-1), all_dists_matched.detach().cpu().numpy()], axis = 0)

      # return distances_all.min(0)
      res_matched = distances_all.min(0)
      
    return res, res_matched
  return res

def compute_overall_singlecd_param(pred_data, input_pointcloud, patch_idx_filter):
  #input pointcloud is a numpy
  #firstly, the grid parts
  # print('matched computation')
  patch_close_logits = pred_data['closed_patch_logits'][0].detach().cpu().numpy()
  patch_uclosed = patch_close_logits[:,0] < patch_close_logits[:,1]
  
  # if args.eval_matched:
  distances, distances_matched = compute_overall_singlecd(pred_data['pred_patch_points'][0].detach().cpu().numpy(), patch_uclosed, input_pointcloud, patch_idx_filter)

  # print('distance shape: ', distances.shape)
  #param part
  cp_distance = ComputePrimitiveDistance(reduce = False)
  routines = {
            5: cp_distance.distance_from_sphere,
            0: cp_distance.distance_from_cylinder,
            4: cp_distance.distance_from_cone,
            3: cp_distance.distance_from_plane,
          }

  src_patch_points = pred_data['pred_patch_points'][0]
  src_with_param = pred_data['pred_patch_with_param'][0]
  src_type_logits = pred_data['pred_patch_type'][0]
  src_param = pred_data['pred_patch_param'][0]
  input_pointcloud_torch = torch.tensor(input_pointcloud, device = src_patch_points.device)
  # print('input pc shape: ',input_pointcloud_torch.shape)
  all_dists = []
  all_dists_matched = []
  # set_match_id = set(matching_indices['indices'][0][0].tolist())
  set_match_id = patch_idx_filter
  print('all patch size: {} matched size: {}'.format(len(src_patch_points), len(set_match_id)))
  for patch_idx in range(len(src_patch_points)):
    if args.eval_param and src_with_param[patch_idx] > 0.5:
      para_dist = routines[torch.argmax(src_type_logits[patch_idx]).item()](input_pointcloud_torch[:,:3], src_param[patch_idx], sqrt = True)
      all_dists.append(para_dist.view([1, -1]))
      if args.eval_matched and patch_idx in set_match_id:
        all_dists_matched.append(para_dist.view([1, -1]))
  
  res = distances
  if len(all_dists) > 0:
    all_dists = torch.cat(all_dists, axis = 0)
    # print('all_dists_shape: ', all_dists.shape)
    distances_all = np.concatenate([distances.reshape(1,-1), all_dists.detach().cpu().numpy()], axis = 0)

    # return distances_all.min(0)
    res = distances_all.min(0)
  
  if args.eval_matched:
    res_matched = distances_matched
    if len(all_dists_matched) > 0:
      all_dists_matched = torch.cat(all_dists_matched, axis = 0)
      # print('all_dists_shape: ', all_dists.shape)
      distances_all = np.concatenate([distances_matched.reshape(1,-1), all_dists_matched.detach().cpu().numpy()], axis = 0)

      # return distances_all.min(0)
      res_matched = distances_all.min(0)
      
    return res, res_matched
  return res

def curve_type_to_id(str):
  #Circle, BSpline, Line, Ellipse
  if(str == 'Circle'):
    return 0
  if(str == 'BSpline'):
    return 1
  if(str == 'Line'):
    return 2
  assert(str == 'Ellipse')
  return 3

def patch_type_to_id(str):
  #Cylinder, Torus, BSpline, Plane, Cone, Sphere
  #update 1011, add Extrusion and Revolution, label same as BSpline
  if(str == 'Cylinder'):
    return 0
  if(str == 'Torus'):
    return 1
  if(str == 'BSpline' or str == 'Extrusion' or str == 'Revolution'):
    return 2
  if(str == 'Plane'):
    return 3
  if(str == 'Cone'):
    return 4
  assert(str == 'Sphere')
  return 5


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [100.0 * torch.ones([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Sparse_Backbone_Minkowski(ME.MinkowskiNetwork):
  def __init__(self):
    super(Sparse_Backbone_Minkowski, self).__init__(3) #dimension = 3
    if(args.backbone_feature_encode): #false
      self.sparseModel = resnets.ResFieldNet34(in_channels=3, out_channels=m*6, D=3)
    elif args.ourresnet:
      if not args.backbone_bn:
        self.sparseModel = resnets.ResNetOur(in_channels=7 if args.input_normal_signals else 4, out_channels=m*6, D=3,flag_expand = args.backbone_expand) #position, normal, 1
      else:
        self.sparseModel = resnets.ResNetOurBn(in_channels=7 if args.input_normal_signals else 4, out_channels=m*6, D=3) #position, normal, 1

    else:
      self.sparseModel = resnets.ResNet34(in_channels=7 if args.input_normal_signals else 4, out_channels=m*6, D=3) #position, normal, 1
    
  def forward(self,x):
    input = ME.SparseTensor(features=x[1], coordinates=x[0])
    #print(input.F)
    #print(input.C)
    out=self.sparseModel(input)
    return out

class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, voxel_dim, num_pos_feats=32, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.voxel_dim = voxel_dim
        assert(voxel_dim > 0)
    
    def forward(self, voxel_coord):
        if self.normalize:
            voxel_coord = self.scale * voxel_coord / (self.voxel_dim - 1)

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=voxel_coord.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos = voxel_coord[:, :, None] / dim_t
        #print("pos.shape: {}".format(pos.shape))
        pos_x = pos[:, 0]
        pos_y = pos[:, 1]
        pos_z = pos[:, 2]#in shape[n, pos_feature_dim]
        #print("pos_x.shape: {}".format(pos_x.shape))
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=2).flatten(1)
        pos = torch.cat((pos_x, pos_y, pos_z), dim=1)#.permute(0, 3, 1, 2)
        #print("final pos.shape: {}".format(pos.shape))
        #print("PositionEmbeddingSine3D:pos:{}".format(pos.device))
        return pos

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sin=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sin_activation = sin

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if(self.sin_activation):
              x = layer(x).sin() if i < self.num_layers - 1 else layer(x)
            else:
              x = F.leaky_relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

hn_hidden_dim = 128
flag_hidden_layer = True

class MLP_hn(nn.Module): #hypernets of MLP
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, input_dim_fea):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        h_plus = [hidden_dim + 1] * (num_layers - 1)
        self.layers_dims = list(zip([input_dim + 1] + h_plus, h + [output_dim]))
        self.layers_size = [a * b for a,b in self.layers_dims]
        #ori version
        if not flag_hidden_layer:
            self.layer = nn.Linear(input_dim_fea, sum(self.layers_size))
        else:
            self.layer1 = nn.Linear(input_dim_fea, hn_hidden_dim)
            self.layer2 = nn.Linear(hn_hidden_dim, sum(self.layers_size))
        #weight change, no correct
        # self.layer.weight.data *= 1.e-2
        # self.layer.bias.data *= 1.e-2
        
    def forward(self, x, feature):
      # print('feature size:', feature.shape)
      # assert(feature)

      #ori version
      if not flag_hidden_layer:
        net_par = self.layer(feature)
      #new version
      else:
        net_par = self.layer1(feature)
        net_par = F.relu(net_par)
        net_par = self.layer2(net_par)
      # #reduce par
      # # net_par = net_par * 1.e-2
      if args.hn_scale:
        net_par = net_par / math.sqrt(hn_pe_dim)

      net_par_layers = torch.split(net_par, self.layers_size, dim=-1)
      for i in range(len(self.layers_size)):
        layer_par = net_par_layers[i].view(net_par.shape[0], net_par.shape[1], net_par.shape[2] ,self.layers_dims[i][0], self.layers_dims[i][1])
        #linear layer
        # x = torch.mm(x, layer_par[:-1]) + layer_par[-1]
        x = torch.einsum('...ij,...jk->...ik', x, layer_par[...,:-1,:]) + layer_par[...,-1:,:]
        # print('x shape: ', x.shape)
        # print('layer par', layer_par.shape)
        # x = torch.einsum('...ij,...jk->...ik', x, layer_par[...,:-1,:])
        if i < self.num_layers - 1:
          x = F.leaky_relu(x)
          # x = F.relu(x)
      return x
        

@torch.no_grad()
def get_attention_mask(voxel_location, seq_length, tokens_each_sample, n_heads=8):
    #*n_heads
    #mask = torch.ones([args.batch_size, tokens_each_sample, tokens_each_sample], dtype= torch.bool)
    mask_list = []
    position_each_sample = torch.split(voxel_location, tokens_each_sample)
    for i in range(args.batch_size):
        voxel_position_cur_sample = position_each_sample[i].float()
        cur_seq_length = voxel_position_cur_sample.shape[0]
        pairwise_distance = torch.cdist(voxel_position_cur_sample, voxel_position_cur_sample)
        cur_mask = F.pad(pairwise_distance > 3, (0, 0, 0, seq_length-cur_seq_length), value=False)
        cur_mask = F.pad(cur_mask, (0, seq_length-cur_seq_length), value=True)
        #set diagonal to False
        #diagonal_indices = torch.arange(seq_length, dtype=torch.long, device=voxel_location.device)
        #cur_mask[diagonal_indices, diagonal_indices] = torch.zeros([seq_length], dtype=torch.bool, device=voxel_location.device)
        #print(seq_length - cur_mask.sum(dim=-1))
        mask_list.append(cur_mask.view(1, seq_length, seq_length).repeat(n_heads, 1, 1)) #.repeat(n_heads, 1, 1)
    mask = torch.cat(mask_list, 0)#
    return mask


class BackBone2VoxelTokens(nn.Module):
    def __init__(self, backbone, position_encoding):
        super().__init__()
        self.backbone = backbone
        self.position_encoding = position_encoding
    
    def forward(self, locations, features):
        #for minkowski, batch index is in axis-0
        locations_pos = locations[:,:3]
        locations_batch_idx = locations[:,-1:]
        locations = torch.cat([locations_batch_idx, locations_pos], dim=1)
        
        #t0 = time.time()
        output = self.backbone([locations, features])
        #t1 = time.time()
        #print("{}s elapsed for Minkowski BackBone".format(t1-t0))
        
        #t0 = time.time()
        sparse_locations = output.C#.to(device)
        sparse_features = output.F
                
        #Padding voxel features and corner points
        batch_idx = sparse_locations[:,0]#which sample each voxel belongs to
        sparse_locations = sparse_locations[:,1:] // output.tensor_stride[0]
        input_padding_mask = torch.zeros_like(batch_idx, dtype=torch.bool, device=sparse_features.device)
        
        batch_number_samples = []
        for i in range(args.batch_size):
          batch_number_samples.append((batch_idx == i).sum())
        
        pad_dim = max(batch_number_samples)
        
        if(args.using_local_attention):
          #print("seq_length = ", pad_dim)
          attention_mask = get_attention_mask(sparse_locations, pad_dim, batch_number_samples, n_heads=args.nheads) #not used
          #print(attention_mask.shape, torch.nonzero(attention_mask).shape)
        else:
          attention_mask = None
        
        voxel_pos_embedding = self.position_encoding(sparse_locations)
        #print(voxel_pos_embedding.shape, sparse_features.shape)
        #sparse_features += voxel_pos_embedding
        voxel_feature = torch.split(sparse_features, batch_number_samples)
        voxel_pos_embedding = torch.split(voxel_pos_embedding, batch_number_samples)
        input_padding_mask = torch.split(input_padding_mask, batch_number_samples)
        
        batch_voxel_feature = []
        input_padding_mask_list = []
        position_embedding_list = []
        for i in range(args.batch_size):
          batch_voxel_feature.append(nn.functional.pad(voxel_feature[i], (0, 0, 0, pad_dim - voxel_feature[i].shape[0])))
          input_padding_mask_list.append(nn.functional.pad(input_padding_mask[i], (0, pad_dim - voxel_feature[i].shape[0]), value=True))
          position_embedding_list.append(nn.functional.pad(voxel_pos_embedding[i], (0, 0, 0, pad_dim - voxel_feature[i].shape[0])))
        
        voxel_features = torch.stack(batch_voxel_feature, dim=1)
        voxel_features_padding_mask = torch.stack(input_padding_mask_list, dim=0)
        voxel_position_encoding = torch.stack(position_embedding_list, dim=1)
        #print("input feature to transformer shape: {}".format(batch_voxel_feature.shape))
        return voxel_features, voxel_position_encoding, voxel_features_padding_mask, torch.split(torch.cat((batch_idx.view(-1,1), sparse_locations), dim=-1), batch_number_samples), attention_mask

class DETR_Corner(nn.Module):
    """ This is the DETR module that performs geometric primitive detection """
    def __init__(self, transformer, num_queries, aux_loss=False): #num_classes is not used for corner detection #backbone
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.empty_prediction_embed = nn.Linear(hidden_dim, 2)#num_classes + 1, empty or non-empty
        self.corner_position_embed = MLP(hidden_dim, hidden_dim, 3, 3) #was bbox embed
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        #self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.aux_loss = aux_loss
        self.corner_topo_embed = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim if not args.matrix_eigen_similarity else 24, 2)
        
        
    def forward(self, voxel_features, voxel_position_encoding, voxel_features_padding_mask, attention_mask=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """                
        src = voxel_features
        pos = voxel_position_encoding
        mask = voxel_features_padding_mask
        #print(src.device, mask.device, self.query_embed.weight.device, pos.device)
        #t1 = time.time()
        #print("{}s elapsed for prepare data for transformer and encoding position".format(t1-t0))
        
        #t0 = time.time()
        assert mask is not None
        hs = self.transformer(src, mask, self.query_embed.weight, pos, src_attention_mask=attention_mask)[0] #self.input_proj, the first element
        #t1 = time.time()
        #print("{}s elapsed for prepare data for transformer forwarding".format(t1-t0))
        #print("hs.shape=", hs.shape)[6, 2, 100, 192]
        
        #t0 = time.time()
        #outputs_class = self.class_embed(hs)
        outputs_corner_coord = self.corner_position_embed(hs).tanh()*0.5 # [-0.5,0.5]
        outputs_class = self.empty_prediction_embed(hs) #to be consistent with curve and patch type prediction, we treat 0 as non-empty and 1 as empty
        
        if(args.normalize_embed_feature): #false
          output_corner_topo_embedding = F.normalize(self.corner_topo_embed(hs), dim=-1)
        else:
          output_corner_topo_embedding = self.corner_topo_embed(hs)
        
        out = {'pred_logits': outputs_class[-1], 'pred_corner_position': outputs_corner_coord[-1], 'corner_topo_embed': output_corner_topo_embedding[-1]}
        if self.aux_loss:
          out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_corner_coord, output_corner_topo_embedding)
        #t1 = time.time()
        #print("{}s elapsed for class and geometry embedding(auxilary included)".format(t1-t0))
        
        return out, hs[-1]#lastlayer

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, output_corner_topo_embedding):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_corner_position': b, 'corner_topo_embed': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], output_corner_topo_embedding[:-1])]

#added on 20210329, only used for forwarding
flag_using_scale = False

# detr curve
class DETR_Curve(nn.Module):
    """ This is the DETR module that performs geometric primitive detection - Curves """
    def __init__(self, transformer, num_queries, aux_loss=False): #num_classes is not used for corner detection #backbone
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.valid_curve_embed = MLP(hidden_dim, hidden_dim, 2, args.curve_embedding_mlp_layers)
        #nn.Linear(hidden_dim, 5)#num_classes + 1, Circle, BSpline, Line, Ellipse or non-empty
        self.curve_type_prediction_embed = MLP(hidden_dim, hidden_dim, 4, args.curve_embedding_mlp_layers)
        self.closed_curve_embed = MLP(hidden_dim, hidden_dim, 2, 3) #not closed, closed
        self.query_embed = nn.Embedding(num_queries, hidden_dim) #independent query embedding
        #self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.aux_loss = aux_loss
        
        self.curve_start_point_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        self.curve_shape_embed = MLP(hidden_dim + 1, hidden_dim, 3, 6, sin=args.sin) #displacement from start point
        self.curve_topo_embed = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim, 3)

        if flag_using_scale:
          self.curve_shape_scale_embed = MLP(hidden_dim, hidden_dim, 1, 3)
        
        '''
        self.line_embed = MLP(hidden_dim, hidden_dim, 6, 3) #predict start point position, end point position
        self.circle_embed = MLP(hidden_dim, hidden_dim, 11, 3) #predict center of circle, radius, start points of arc(axis=0), angle of arc, axis-1(make orthogonal to axis-0)
        self.ellipse_embed = MLP(hidden_dim, hidden_dim, 7, 3) #predict center of ellipse, axis-0 radius, start(axis-0 direction), axis-1 direction and radius,  end points of arc, angle of arc
        
        self.bSpline_startpoint_embed = MLP(hidden_dim, hidden_dim, 5, 3) #predict start position as well as parameterization space of spline
        self.bSpline_shape_embed = MLP(hidden_dim+1, hidden_dim, 3, 3) #predict shape of spline
        '''
         
    def forward(self, voxel_features, voxel_position_encoding, voxel_features_padding_mask, attention_mask=None, corner_query_features=None, primitive_type_embed=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        #pri embd: 3x192
        src = voxel_features
        pos = voxel_position_encoding
        mask = voxel_features_padding_mask
        #print(src.device, mask.device, self.query_embed.weight.device, pos.device)
        
        if(corner_query_features is not None and primitive_type_embed is not None):
          assert(attention_mask is None)
          #add voxel embedding feature to voxel features
          voxel_features_mod = src + primitive_type_embed[0]
          corner_query_features_mod = corner_query_features + primitive_type_embed[1]
          src = torch.cat([corner_query_features_mod.transpose(0,1), voxel_features_mod], dim=0)
          #print(mask.shape, pos.shape)
          mask = F.pad(mask, (args.num_queries, 0), value=False)
          #TODO: add corner position encoding or query index position encoding
          pos = F.pad(pos, (0, 0, 0, 0, args.num_queries, 0), value=0) #from back to front
          
        
        #t0 = time.time()
        assert mask is not None
        hs = self.transformer(src, mask, self.query_embed.weight, pos, src_attention_mask=attention_mask)[0] #self.input_proj
        #t1 = time.time()
        #print("{}s elapsed for prepare data for transformer forwarding".format(t1-t0))
        #print("hs.shape=", hs.shape)[6, 2, 100, 192]
        
        #t0 = time.time()
        #outputs_class = self.class_embed(hs)
        outputs_start_point_coord = self.curve_start_point_embed(hs).tanh()*0.5 # [-0.5,0.5]
        if not flag_using_scale:
          curve_shape_scale = 1#F.elu(self.curve_shape_scale_embed(hs)).unsqueeze(-2) + 1.0
        else:
          curve_shape_scale = F.elu(self.curve_shape_scale_embed(hs)).unsqueeze(-2) + 1.0
        #assert(len(outputs_start_point_coord.shape) + 1 == len(curve_shape_scale.shape))
        
        parameterization_coord = (torch.arange(points_per_curve, dtype=torch.float32, device=voxel_features.device) / (points_per_curve - 1)).view(1, 1, 1, points_per_curve, 1).repeat(args.dec_layers, args.batch_size, args.num_queries, 1, 1)
        sampled_points_feature = torch.cat([hs.unsqueeze(-2).repeat(1, 1, 1, points_per_curve, 1), parameterization_coord], dim=-1)
        #in shape [6, bs, 100, 34, 3]
        sampled_points = outputs_start_point_coord.unsqueeze(-2).repeat(1,1,1,points_per_curve,1) + curve_shape_scale*self.curve_shape_embed(sampled_points_feature)
        
        is_curve_closed_logits = self.closed_curve_embed(hs)
        
        is_curve_valid_pred = self.valid_curve_embed(hs)
        outputs_class = self.curve_type_prediction_embed(hs) #Circle, BSpline, Line, Ellipse or non-empty
        
        if(args.normalize_embed_feature):
          output_curve_topo_embedding = F.normalize(self.curve_topo_embed(hs), dim=-1)
        else:
          output_curve_topo_embedding = self.curve_topo_embed(hs)
        
        out = {'pred_curve_logits': is_curve_valid_pred[-1], 'pred_curve_type': outputs_class[-1], 'pred_curve_points': sampled_points[-1], 'closed_curve_logits':is_curve_closed_logits[-1], 'curve_topo_embed': output_curve_topo_embedding[-1]}
        if self.aux_loss:
          out['aux_outputs'] = self._set_aux_loss(is_curve_valid_pred, outputs_class, sampled_points, is_curve_closed_logits, output_curve_topo_embedding)
        #t1 = time.time()
        #print("{}s elapsed for class and geometry embedding(auxilary included)".format(t1-t0))
        
        return out, hs[-1]

    @torch.jit.unused
    def _set_aux_loss(self, is_curve_valid_pred, outputs_class, outputs_coord, closed_curve_logits, output_curve_topo_embedding):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_curve_logits': e, 'pred_curve_type': a, 'pred_curve_points': b, 'closed_curve_logits': c, "curve_topo_embed": d}
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1], closed_curve_logits[:-1], output_curve_topo_embedding[:-1], is_curve_valid_pred[:-1])]

class DETR_Patch(nn.Module):
    """ This is the DETR module that performs geometric primitive detection - Curves """
    def __init__(self, transformer, num_queries, aux_loss=False): #num_classes is not used for corner detection #backbone
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.valid_patch_embed = MLP(hidden_dim, hidden_dim, 2, args.curve_embedding_mlp_layers)
        #Cylinder, Torus, BSpline, Plane, Cone, Sphere
        self.patch_type_prediction_embed = MLP(hidden_dim, hidden_dim, 6, args.curve_embedding_mlp_layers)
        self.query_embed = nn.Embedding(num_queries, hidden_dim) #independent query embedding
        self.aux_loss = aux_loss
        
        self.patch_center_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        if flag_using_scale:
          self.patch_shape_scale_embed = MLP(hidden_dim, hidden_dim, 1, 3)
        self.patch_shape_embed = MLP(hidden_dim + 2, hidden_dim, 3, 6, sin=args.sin) #displacement from start point          
        self.patch_topo_embed = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim, 3)
         
    def forward(self, voxel_features, voxel_position_encoding, voxel_features_padding_mask, attention_mask=None, curve_query_features=None, primitive_type_embed=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
                
        src = voxel_features
        pos = voxel_position_encoding
        mask = voxel_features_padding_mask
        #print(src.device, mask.device, self.query_embed.weight.device, pos.device)
        
        if(curve_query_features is not None and primitive_type_embed is not None):
          assert(attention_mask is None)
          #add voxel embedding feature to voxel features
          #src: 300,1,192
          #primitive_type_embed: 3,192
          voxel_features_mod = src + primitive_type_embed[0]
          #curve_query_features: 1, 100, 192
          curve_query_features_mod = curve_query_features + primitive_type_embed[2]
          src = torch.cat([curve_query_features_mod.transpose(0,1), voxel_features_mod], dim=0)
          #print(mask.shape, pos.shape)
          mask = F.pad(mask, (args.num_queries, 0), value=False)
          #TODO: add corner position encoding or query index position encoding
          pos = F.pad(pos, (0, 0, 0, 0, args.num_queries, 0), value=0)
          
        
        #t0 = time.time()
        assert mask is not None
        hs = self.transformer(src, mask, self.query_embed.weight, pos, src_attention_mask=attention_mask)[0] #self.input_proj
        #t1 = time.time()
        #print("{}s elapsed for prepare data for transformer forwarding".format(t1-t0))
        #print("hs.shape=", hs.shape)[6, 2, 100, 192]
        
        #t0 = time.time()
        outputs_start_point_coord = self.patch_center_embed(hs).tanh()*0.5 # [-0.5,0.5]
        if not flag_using_scale:
          patch_shape_scale = 1#F.elu(self.patch_shape_scale_embed(hs)).unsqueeze(-2) + 1.0
        else:
          patch_shape_scale = F.elu(self.patch_shape_scale_embed(hs)).unsqueeze(-2) + 1.0
        parameterization_coord = torch.arange(points_per_patch_dim*points_per_patch_dim, dtype=torch.int32, device=voxel_features.device)
        parameterization_coord = (torch.cat([(parameterization_coord // points_per_patch_dim).view(-1,1), (parameterization_coord % points_per_patch_dim).view(-1,1)], dim=1).float() / (points_per_patch_dim - 1)).view(1, 1, 1, points_per_patch_dim*points_per_patch_dim, 2).repeat(args.dec_layers, args.batch_size, args.num_queries, 1, 1)
        sampled_points_feature = torch.cat([hs.unsqueeze(-2).repeat(1, 1, 1, points_per_patch_dim*points_per_patch_dim, 1), parameterization_coord], dim=-1)
        #in shape [6, bs, 100, 100*100, 3]
        sampled_points = outputs_start_point_coord.unsqueeze(-2).repeat(1,1,1,points_per_patch_dim*points_per_patch_dim,1) + patch_shape_scale*self.patch_shape_embed(sampled_points_feature)
        
        is_patch_valid_pred = self.valid_patch_embed(hs)
        outputs_class = self.patch_type_prediction_embed(hs) #Circle, BSpline, Line, Ellipse or non-empty
        
        if(args.normalize_embed_feature):
          output_patch_topo_embedding = F.normalize(self.patch_topo_embed(hs), dim=-1)
        else:
          output_patch_topo_embedding = self.patch_topo_embed(hs)
        
        out = {'pred_patch_logits': is_patch_valid_pred[-1], 'pred_patch_type': outputs_class[-1], 'pred_patch_points': sampled_points[-1], 'patch_topo_embed': output_patch_topo_embedding[-1]}
        if self.aux_loss:
          out['aux_outputs'] = self._set_aux_loss(is_patch_valid_pred, outputs_class, sampled_points, output_patch_topo_embedding)
        #t1 = time.time()
        #print("{}s elapsed for class and geometry embedding(auxilary included)".format(t1-t0))
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, is_patch_valid_pred, outputs_class, outputs_coord, output_patch_topo_embedding):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_patch_logits': e, 'pred_patch_type': a, 'pred_patch_points': b, "patch_topo_embed": d}
                for a, b, d, e in zip(outputs_class[:-1], outputs_coord[:-1], output_patch_topo_embedding[:-1], is_patch_valid_pred[:-1])]

class DETR_Shape(nn.Module):
    def __init__(self, backbone, position_encoding, corner_transformer, curve_transformer, patch_transformer, num_queries, aux_loss=False):
        super().__init__()
        self.backbone = BackBone2VoxelTokens(backbone, position_encoding)
        self.corner_model = DETR_Corner(corner_transformer, num_queries, aux_loss)
        self.curve_model = DETR_Curve(curve_transformer, num_queries, aux_loss)
        self.patch_model = DETR_Patch(patch_transformer, num_queries, aux_loss)
        self.primitive_type_embed = nn.Embedding(3, m*6) #voxel, corner, curve
        # self.primitive_type_embed = nn.Embedding(4, m*6) #voxel, corner, curve

    
    def forward(self, locations, features):
        voxel_features, voxel_position_encoding, voxel_features_padding_mask, sparse_locations, attention_mask = self.backbone(locations, features) #feature: pos
        corner_predictions, corner_query_features = self.corner_model(voxel_features, voxel_position_encoding, voxel_features_padding_mask, attention_mask)
        #print("corner_query_features.shape =", corner_query_features.shape) #in shape [2, 100, 192]
        #print("voxel_features.shape = ", voxel_features.shape, " voxel_position_encoding.shape", voxel_position_encoding.shape) [1598, 2, 192]
        #print(self.primitive_type_embed.weight)
        curve_predictions, curve_query_features = self.curve_model(voxel_features, voxel_position_encoding, voxel_features_padding_mask, attention_mask, corner_query_features=corner_query_features, primitive_type_embed=self.primitive_type_embed.weight)
        patch_predictions                       = self.patch_model(voxel_features, voxel_position_encoding, voxel_features_padding_mask, attention_mask, curve_query_features=curve_query_features, primitive_type_embed=self.primitive_type_embed.weight)
        return sparse_locations, corner_predictions, curve_predictions, patch_predictions


class DETR_Corner_Tripath(nn.Module):
    """ This is the DETR module that performs geometric primitive detection """
    def __init__(self, num_queries, hidden_dim, aux_loss=False): #num_classes is not used for corner detection #backbone
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        # hidden_dim = transformer.d_model
        self.empty_prediction_embed = nn.Linear(hidden_dim, 2)#num_classes + 1, empty or non-empty
        self.corner_position_embed = MLP(hidden_dim, hidden_dim, 3, 3) #was bbox embed
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        #self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.aux_loss = aux_loss
        
        if not args.no_topo:
          if args.ori_topo:
            self.corner_topo_embed = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim if not args.matrix_eigen_similarity else 24, 2) #topo embed dim:256, matrix_eigen_similarity: False
          else:
            self.corner_topo_embed_curve = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim, 1) #topo embed dim:256, matrix_eigen_similarity: False
            self.corner_topo_embed_patch = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim, 1) #topo embed dim:256, matrix_eigen_similarity: False
          #(self, input_dim, hidden_dim, output_dim, num_layers, sin=False):
          
        
    def forward(self, hs):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """                
        # src = voxel_features
        # pos = voxel_position_encoding
        # mask = voxel_features_padding_mask
        # #print(src.device, mask.device, self.query_embed.weight.device, pos.device)
        # #t1 = time.time()
        # #print("{}s elapsed for prepare data for transformer and encoding position".format(t1-t0))
        
        # #t0 = time.time()
        # assert mask is not None
        # hs = self.transformer(src, mask, self.query_embed.weight, pos, src_attention_mask=attention_mask)[0] #self.input_proj, the first element
        #t1 = time.time()
        #print("{}s elapsed for prepare data for transformer forwarding".format(t1-t0))
        #print("hs.shape=", hs.shape)[6, 2, 100, 192]
        
        #t0 = time.time()
        #outputs_class = self.class_embed(hs)
        outputs_corner_coord = self.corner_position_embed(hs).tanh()*0.5 # [-0.5,0.5] #6,1,100,3
        outputs_class = self.empty_prediction_embed(hs) #to be consistent with curve and patch type prediction, we treat 0 as non-empty and 1 as empty
        
        if not args.no_topo:
          if(args.normalize_embed_feature):
            output_corner_topo_embedding = F.normalize(self.corner_topo_embed(hs), dim=-1)
          else:
            if args.ori_topo:
              output_corner_topo_embedding = self.corner_topo_embed(hs)
            else:
              output_corner_topo_embedding_curve = self.corner_topo_embed_curve(hs)
              output_corner_topo_embedding_patch = self.corner_topo_embed_patch(hs)
        
        if not args.no_topo:
          if args.ori_topo:
            out = {'pred_logits': outputs_class[-1], 'pred_corner_position': outputs_corner_coord[-1], 'corner_topo_embed': output_corner_topo_embedding[-1]} #only return last layer info
          else:
            out = {'pred_logits': outputs_class[-1], 'pred_corner_position': outputs_corner_coord[-1], 'corner_topo_embed_curve': output_corner_topo_embedding_curve[-1], 'corner_topo_embed_patch': output_corner_topo_embedding_patch[-1]} #only return last layer info         
          if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_corner_coord, output_corner_topo_embedding)
        else:
          out = {'pred_logits': outputs_class[-1], 'pred_corner_position': outputs_corner_coord[-1]} #only return last layer info
        #t1 = time.time()
        #print("{}s elapsed for class and geometry embedding(auxilary included)".format(t1-t0))
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, output_corner_topo_embedding):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_corner_position': b, 'corner_topo_embed': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], output_corner_topo_embedding[:-1])]

# detr curve
flag_hn_with_coord = True
class DETR_Curve_Tripath(nn.Module):
    """ This is the DETR module that performs geometric primitive detection - Curves """
    def __init__(self, num_queries, hidden_dim,  aux_loss=False, device = None): #num_classes is not used for corner detection #backbone
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        # self.transformer = transformer
        # hidden_dim = transformer.d_model
        self.valid_curve_embed = MLP(hidden_dim, hidden_dim, 2, args.curve_embedding_mlp_layers)
        #nn.Linear(hidden_dim, 5)#num_classes + 1, Circle, BSpline, Line, Ellipse or non-empty
        self.curve_type_prediction_embed = MLP(hidden_dim, hidden_dim, 4, args.curve_embedding_mlp_layers)
        self.closed_curve_embed = MLP(hidden_dim, hidden_dim, 2, 3) #not closed, closed
        # self.query_embed = nn.Embedding(num_queries, hidden_dim) #independent query embedding
        #self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.aux_loss = aux_loss
        
        self.curve_start_point_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        
        if args.no_pe:
          if True:
            #no pe
            self.curve_shape_embed = MLP_hn(1, hn_mlp_dim, 3, 3, hidden_dim)
            # coord = torch.arange()
            # coord = (torch.arange(points_per_curve, dtype=torch.float32, device=device) / (points_per_curve - 1)).view(-1,1)
            # exp  = torch.arange(hn_pe_dim//2, dtype=torch.float32, device=device)
            # base = args.pe_sin_base * torch.ones([hn_pe_dim//2], dtype=torch.float32, device=device)
            # coeff = 2 * math.pi * torch.pow(base, exp).view(1,-1)
            # mat = torch.mm(coord, coeff)
            # sin_mat = torch.sin(mat).view(-1,1)
            # cos_mat = torch.cos(mat).view(-1,1)
            # self.curve_pe = torch.cat([sin_mat, cos_mat], dim=1).view(points_per_curve, hn_pe_dim)
            self.curve_pe = (torch.arange(points_per_curve, dtype=torch.float32, device=device) / (points_per_curve - 1)).view(points_per_curve,1)
            # self.curve_pe = torch.zeros([points_per_curve, hn_pe_dim], dtype=torch.float32, device = device)
        elif not args.pe_sin:
          if args.ori_mlp:
            self.curve_shape_embed = MLP(hidden_dim + 1, hidden_dim, 3, 6, sin=args.sin) #ori version
            # self.curve_shape_embed = MLP(hidden_dim + 1, args.hidden_dim_mlp, 3, 6, sin=args.sin)
            # self.curve_shape_embed = MLP(hidden_dim + 1, args.hidden_dim_mlp, 3, 5, sin=args.sin)

            # self.curve_shape_embed = MLP(hidden_dim + 1, hidden_dim, 3, 8, sin=args.sin) #1228 #displacement from start point
            #displacement from start point
          else:
            if flag_hn_with_coord:
              self.curve_shape_embed = MLP_hn(hn_pe_dim + 1, hn_mlp_dim, 3, 3, hidden_dim)
            else:
              self.curve_shape_embed = MLP_hn(hn_pe_dim, hn_mlp_dim, 3, 3, hidden_dim)
            self.curve_pe = nn.Embedding(points_per_curve, hn_pe_dim)
        else:
          self.curve_shape_embed = MLP_hn(hn_pe_dim, hn_mlp_dim, 3, 3, hidden_dim)
          # coord = torch.arange()
          coord = (torch.arange(points_per_curve, dtype=torch.float32, device=device) / (points_per_curve - 1)).view(-1,1)
          exp  = torch.arange(hn_pe_dim//2, dtype=torch.float32, device=device)
          base = args.pe_sin_base * torch.ones([hn_pe_dim//2], dtype=torch.float32, device=device)
          coeff = 2 * math.pi * torch.pow(base, exp).view(1,-1)
          mat = torch.mm(coord, coeff)
          sin_mat = torch.sin(mat).view(-1,1)
          cos_mat = torch.cos(mat).view(-1,1)
          self.curve_pe = torch.cat([sin_mat, cos_mat], dim=1).view(points_per_curve, hn_pe_dim)
          # self.curve_pe = torch.zeros([points_per_curve, hn_pe_dim], dtype=torch.float32, device = device)

          
        if not args.no_topo:
          if args.ori_topo:
            self.curve_topo_embed = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim, 3)
          else:
            self.curve_topo_embed_corner = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim, 1)
            self.curve_topo_embed_patch = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim, 1)

        if flag_using_scale:
          self.curve_shape_scale_embed = MLP(hidden_dim, hidden_dim, 1, 3)
        
        '''
        self.line_embed = MLP(hidden_dim, hidden_dim, 6, 3) #predict start point position, end point position
        self.circle_embed = MLP(hidden_dim, hidden_dim, 11, 3) #predict center of circle, radius, start points of arc(axis=0), angle of arc, axis-1(make orthogonal to axis-0)
        self.ellipse_embed = MLP(hidden_dim, hidden_dim, 7, 3) #predict center of ellipse, axis-0 radius, start(axis-0 direction), axis-1 direction and radius,  end points of arc, angle of arc
        
        self.bSpline_startpoint_embed = MLP(hidden_dim, hidden_dim, 5, 3) #predict start position as well as parameterization space of spline
        self.bSpline_shape_embed = MLP(hidden_dim+1, hidden_dim, 3, 3) #predict shape of spline
        '''
         
    def forward(self, hs):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # #pri embd: 3x192
        # src = voxel_features
        # pos = voxel_position_encoding
        # mask = voxel_features_padding_mask
        # #print(src.device, mask.device, self.query_embed.weight.device, pos.device)
        
        # if(corner_query_features is not None and primitive_type_embed is not None):
        #   assert(attention_mask is None)
        #   #add voxel embedding feature to voxel features
        #   voxel_features_mod = src + primitive_type_embed[0]
        #   corner_query_features_mod = corner_query_features + primitive_type_embed[1]
        #   src = torch.cat([corner_query_features_mod.transpose(0,1), voxel_features_mod], dim=0)
        #   #print(mask.shape, pos.shape)
        #   mask = F.pad(mask, (args.num_queries, 0), value=False)
        #   #TODO: add corner position encoding or query index position encoding
        #   pos = F.pad(pos, (0, 0, 0, 0, args.num_queries, 0), value=0)
          
        
        # #t0 = time.time()
        # assert mask is not None
        # hs = self.transformer(src, mask, self.query_embed.weight, pos, src_attention_mask=attention_mask)[0] #self.input_proj
        #t1 = time.time()
        #print("{}s elapsed for prepare data for transformer forwarding".format(t1-t0))
        #print("hs.shape=", hs.shape)[6, 2, 100, 192] #six layer
        
        #t0 = time.time()
        #outputs_class = self.class_embed(hs)
        outputs_start_point_coord = self.curve_start_point_embed(hs).tanh()*0.5 # [-0.5,0.5]
        if not flag_using_scale:
          curve_shape_scale = 1#F.elu(self.curve_shape_scale_embed(hs)).unsqueeze(-2) + 1.0
        else:
          curve_shape_scale = F.elu(self.curve_shape_scale_embed(hs)).unsqueeze(-2) + 1.0
        #assert(len(outputs_start_point_coord.shape) + 1 == len(curve_shape_scale.shape))
        
        # if True:
        if args.enable_aux_loss:
          parameterization_coord = (torch.arange(points_per_curve, dtype=torch.float32, device=hs.device) / (points_per_curve - 1)).view(1, 1, 1, points_per_curve, 1).repeat(args.dec_layers, args.batch_size, args.num_curve_queries, 1, 1)
        else:
          parameterization_coord = (torch.arange(points_per_curve, dtype=torch.float32, device=hs.device) / (points_per_curve - 1)).view(1, 1, 1, points_per_curve, 1).repeat(1, args.batch_size, args.num_curve_queries, 1, 1)
        
        if args.no_pe:
          if True:
            sampled_points_feature = self.curve_pe.view(1,1,1,points_per_curve, 1).repeat(1, args.batch_size, args.num_curve_queries, 1,1)
            sampled_points = outputs_start_point_coord.unsqueeze(-2).repeat(1,1,1,points_per_curve,1) + self.curve_shape_embed(sampled_points_feature, hs)
        elif not args.pe_sin:
          if args.ori_mlp:
            # sampled_points_feature = torch.cat([hs.unsqueeze(-2).repeat(1, 1, 1, points_per_curve, 1), parameterization_coord], dim=-1)
            # #in shape [6, bs, 100, 34, 3]
            # sampled_points = outputs_start_point_coord.unsqueeze(-2).repeat(1,1,1,points_per_curve,1) + curve_shape_scale*self.curve_shape_embed(sampled_points_feature)
            sampled_points_feature = torch.cat([hs.unsqueeze(-2).repeat(1, 1, 1, points_per_curve, 1), parameterization_coord], dim=-1)
            #in shape [6, bs, 100, 34, 3]
            sampled_points = outputs_start_point_coord.unsqueeze(-2).repeat(1,1,1,points_per_curve,1) + curve_shape_scale*self.curve_shape_embed(sampled_points_feature)
          else:
            # sampled_points_feature = torch.cat([self.curve_pe.weight.view(1,1,1,points_per_curve, hn_pe_dim).repeat(1, args.batch_size, args.num_queries, 1,1) ,parameterization_coord], dim=-1)
            
            #not concate coord
            #hs shape: [1, bs, 100, *]
            if flag_hn_with_coord:
              sampled_points_feature = torch.cat([self.curve_pe.weight.view(1,1,1,points_per_curve, hn_pe_dim).repeat(1, args.batch_size, args.num_curve_queries, 1,1), parameterization_coord], dim=-1)
            else:
              sampled_points_feature = self.curve_pe.weight.view(1,1,1,points_per_curve, hn_pe_dim).repeat(1, args.batch_size, args.num_curve_queries, 1,1)
            sampled_points = outputs_start_point_coord.unsqueeze(-2).repeat(1,1,1,points_per_curve,1) + self.curve_shape_embed(sampled_points_feature, hs)
        else:
          sampled_points_feature = self.curve_pe.view(1,1,1,points_per_curve, hn_pe_dim).repeat(1, args.batch_size, args.num_curve_queries, 1,1)
          sampled_points = outputs_start_point_coord.unsqueeze(-2).repeat(1,1,1,points_per_curve,1) + self.curve_shape_embed(sampled_points_feature, hs)


        is_curve_closed_logits = self.closed_curve_embed(hs)
        
        is_curve_valid_pred = self.valid_curve_embed(hs)
        outputs_class = self.curve_type_prediction_embed(hs) #Circle, BSpline, Line, Ellipse or non-empty
        
        if not args.no_topo:
          if(args.normalize_embed_feature):
            output_curve_topo_embedding = F.normalize(self.curve_topo_embed(hs), dim=-1)
          else:
            if args.ori_topo:
              output_curve_topo_embedding = self.curve_topo_embed(hs)
            else:
              output_curve_topo_embedding_corner = self.curve_topo_embed_corner(hs)
              output_curve_topo_embedding_patch = self.curve_topo_embed_patch(hs)
              # output_curve_topo_embedding_patch = self.curve_topo_embed_corner(hs)
              
        if not args.no_topo:
          if args.ori_topo:
            out = {'pred_curve_logits': is_curve_valid_pred[-1], 'pred_curve_type': outputs_class[-1], 'pred_curve_points': sampled_points[-1], 'closed_curve_logits':is_curve_closed_logits[-1], 'curve_topo_embed': output_curve_topo_embedding[-1]}
          else:
            out = {'pred_curve_logits': is_curve_valid_pred[-1], 'pred_curve_type': outputs_class[-1], 'pred_curve_points': sampled_points[-1], 'closed_curve_logits':is_curve_closed_logits[-1], 'curve_topo_embed_corner': output_curve_topo_embedding_corner[-1], 'curve_topo_embed_patch': output_curve_topo_embedding_patch[-1]}
          if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(is_curve_valid_pred, outputs_class, sampled_points, is_curve_closed_logits, output_curve_topo_embedding)
        else:
           out = {'pred_curve_logits': is_curve_valid_pred[-1], 'pred_curve_type': outputs_class[-1], 'pred_curve_points': sampled_points[-1], 'closed_curve_logits':is_curve_closed_logits[-1]}
        #t1 = time.time()
        #print("{}s elapsed for class and geometry embedding(auxilary included)".format(t1-t0))
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, is_curve_valid_pred, outputs_class, outputs_coord, closed_curve_logits, output_curve_topo_embedding):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_curve_logits': e, 'pred_curve_type': a, 'pred_curve_points': b, 'closed_curve_logits': c, "curve_topo_embed": d}
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1], closed_curve_logits[:-1], output_curve_topo_embedding[:-1], is_curve_valid_pred[:-1])]

class DETR_Patch_Tripath(nn.Module):
    """ This is the DETR module that performs geometric primitive detection - Curves """
    def __init__(self, num_queries,hidden_dim,  aux_loss=False, device = None): #num_classes is not used for corner detection #backbone
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        # self.transformer = transformer
        # hidden_dim = transformer.d_model
        self.valid_patch_embed = MLP(hidden_dim, hidden_dim, 2, args.curve_embedding_mlp_layers)
        #Cylinder, Torus, BSpline, Plane, Cone, Sphere
        self.patch_type_prediction_embed = MLP(hidden_dim, hidden_dim, 6, args.curve_embedding_mlp_layers)
        if args.patch_close:
          self.closed_patch_embed = MLP(hidden_dim, hidden_dim, 2, 3) #not closed, closed

        # self.query_embed = nn.Embedding(num_queries, hidden_dim) #independent query embedding
        self.aux_loss = aux_loss
        
        self.patch_center_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        if flag_using_scale:
          self.patch_shape_scale_embed = MLP(hidden_dim, hidden_dim, 1, 3)
        # self.patch_shape_embed = MLP(hidden_dim + 2, hidden_dim, 3, 6, sin=args.sin) #displacement from start point
        output_dim = 3
        if args.output_normal:
          output_dim = 6

        if args.no_pe:
          self.patch_shape_embed = MLP_hn(2, hn_mlp_dim, output_dim, 3, hidden_dim)
          self.patch_pe_x = (torch.arange(points_per_patch_dim, dtype=torch.float32, device=device) / (points_per_patch_dim - 1)).view(points_per_patch_dim,1)
          self.patch_pe_y = (torch.arange(points_per_patch_dim, dtype=torch.float32, device=device) / (points_per_patch_dim - 1)).view(points_per_patch_dim,1)
        elif args.spe:
          self.patch_shape_embed = MLP_hn(6, hn_mlp_dim, output_dim, 3, hidden_dim)
          coordx = (torch.arange(points_per_patch_dim, dtype=torch.float32, device=device) / (points_per_patch_dim - 1)).view(points_per_patch_dim,1)
          sin_mat_x = torch.sin(2 * math.pi * coordx).view(-1,1)
          cos_mat_x = torch.cos(2 * math.pi * coordx).view(-1,1)
          self.patch_pe_x = torch.cat([coordx, sin_mat_x, cos_mat_x], dim=1).view(points_per_patch_dim, 3)
          coordy = (torch.arange(points_per_patch_dim, dtype=torch.float32, device=device) / (points_per_patch_dim - 1)).view(points_per_patch_dim,1)
          sin_mat_y = torch.sin(2 * math.pi * coordy).view(-1,1)
          cos_mat_y = torch.cos(2 * math.pi * coordy).view(-1,1)
          self.patch_pe_y = torch.cat([coordy, sin_mat_y, cos_mat_y], dim=1).view(points_per_patch_dim, 3)

        elif not args.pe_sin:
          if args.ori_mlp:
            # self.patch_shape_embed = MLP(hidden_dim + 2, args.hidden_dim_mlp, 3, 6, sin=args.sin) 
            # self.patch_shape_embed = MLP(hidden_dim + 2, args.hidden_dim_mlp, 3, 5, sin=args.sin) 

            self.patch_shape_embed = MLP(hidden_dim + 2, hidden_dim, 3, 6, sin=args.sin) #1227, ori
            #displacement from start point
          else:
            if flag_hn_with_coord:
              self.patch_shape_embed = MLP_hn(hn_pe_dim * 2 + 2, hn_mlp_dim, output_dim, 3, hidden_dim)
            else:
              self.patch_shape_embed = MLP_hn(hn_pe_dim * 2, hn_mlp_dim, output_dim, 3, hidden_dim)
            self.patch_pe_x = nn.Embedding(points_per_patch_dim, hn_pe_dim)
            self.patch_pe_y = nn.Embedding(points_per_patch_dim, hn_pe_dim)
        else:
          self.patch_shape_embed = MLP_hn(hn_pe_dim * 2, hn_mlp_dim, 3, 3, hidden_dim)
          #for computing patch_pe_x
          coord = (torch.arange(points_per_patch_dim, dtype=torch.float32, device=device) / (points_per_patch_dim - 1)).view(-1,1)
          exp  = torch.arange(hn_pe_dim//2, dtype=torch.float32, device=device)
          base = args.pe_sin_base * torch.ones([hn_pe_dim//2], dtype=torch.float32, device=device)
          coeff = 2 * math.pi * torch.pow(base, exp).view(1,-1)
          mat = torch.mm(coord, coeff)
          sin_mat = torch.sin(mat).view(-1,1)
          cos_mat = torch.cos(mat).view(-1,1)
          # self.curve_pe = torch.cat([sin_mat, cos_mat], dim=1).view(points_per_curve, hn_pe_dim)
          self.patch_pe_x = torch.cat([sin_mat, cos_mat], dim=1).view(points_per_patch_dim, hn_pe_dim)
          self.patch_pe_y = self.patch_pe_x

        if not args.no_topo:
          if args.ori_topo:
            self.patch_topo_embed = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim, 3)
          else:
            self.patch_topo_embed_curve = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim, 1)
            self.patch_topo_embed_corner = MLP(hidden_dim, args.topo_embed_dim, args.topo_embed_dim, 1)
        
        if args.patch_normal or args.output_normal:
          mask = torch.arange(points_per_patch_dim*points_per_patch_dim, dtype=torch.int64, device=device).view(points_per_patch_dim, points_per_patch_dim)
          self.mask_x = torch.cat([mask[:,1:], mask[:, -2:-1]], dim = 1).view(points_per_patch_dim*points_per_patch_dim)
          self.mask_y = torch.cat([mask[1:], mask[-2:-1]], dim = 0).view(points_per_patch_dim*points_per_patch_dim)
          
        if args.patch_lap:
          mask = torch.arange(points_per_patch_dim*points_per_patch_dim, dtype=torch.int64, device=device).view(points_per_patch_dim, points_per_patch_dim)
          # self.mask_x_minus = torch.cat([mask[:,0:1], mask[:,0:-2], mask[:,-1:]], dim = 1).view(points_per_patch_dim*points_per_patch_dim)
          # self.mask_x_plus = torch.cat([mask[:,0:1], mask[:,2:], mask[:,-1:]], dim = 1).view(points_per_patch_dim*points_per_patch_dim)
          # self.mask_y_minus = torch.cat([mask[0:1], mask[0:-2], mask[-1:]], dim = 0).view(points_per_patch_dim*points_per_patch_dim)
          # self.mask_y_plus = torch.cat([mask[0:1], mask[2:], mask[-1:]], dim = 0).view(points_per_patch_dim*points_per_patch_dim)

          self.mask_x_minus = torch.arange(points_per_patch_dim*points_per_patch_dim, dtype=torch.int64, device=device).view(points_per_patch_dim, points_per_patch_dim)
          self.mask_x_plus = torch.arange(points_per_patch_dim*points_per_patch_dim, dtype=torch.int64, device=device).view(points_per_patch_dim, points_per_patch_dim)
          self.mask_y_minus = torch.arange(points_per_patch_dim*points_per_patch_dim, dtype=torch.int64, device=device).view(points_per_patch_dim, points_per_patch_dim)
          self.mask_y_plus = torch.arange(points_per_patch_dim*points_per_patch_dim, dtype=torch.int64, device=device).view(points_per_patch_dim, points_per_patch_dim)

          self.mask_x_minus[1:-1,1:-1] = mask[1:-1,:-2]
          self.mask_x_plus[1:-1,1:-1] = mask[1:-1,2:]
          self.mask_y_minus[1:-1,1:-1] = mask[:-2, 1:-1]
          self.mask_y_plus[1:-1,1:-1] = mask[2:, 1:-1]

          self.mask_x_minus = self.mask_x_minus.view(points_per_patch_dim*points_per_patch_dim)
          self.mask_x_plus = self.mask_x_plus.view(points_per_patch_dim*points_per_patch_dim)
          self.mask_y_minus = self.mask_y_minus.view(points_per_patch_dim*points_per_patch_dim)
          self.mask_y_plus = self.mask_y_plus.view(points_per_patch_dim*points_per_patch_dim)

        if args.patch_lapboundary:
          self.mat_lapboundary = torch.zeros([points_per_patch_dim*points_per_patch_dim, points_per_patch_dim*points_per_patch_dim], dtype = torch.float32, device = device)
          four_nb = np.array([[-1,0], [1, 0], [0,1], [0,-1]])
          for i in range(points_per_patch_dim):
            for j in range(points_per_patch_dim):
              cur_id = i * points_per_patch_dim + j
              cur_id_array = np.array([i, j])
              for k in range(4):
                tmp_id_array = cur_id_array + four_nb[k]
                if tmp_id_array[0] < points_per_patch_dim and tmp_id_array[0] >= 0 and tmp_id_array[1] < points_per_patch_dim and tmp_id_array[1] >= 0:
                  self.mat_lapboundary[cur_id, tmp_id_array[0] * points_per_patch_dim + tmp_id_array[1]] = 1.0
          
          self.mat_lapboundary = self.mat_lapboundary / self.mat_lapboundary.sum(dim = -1).unsqueeze(-1)
          self.mat_lapboundary = torch.eye(points_per_patch_dim*points_per_patch_dim, device = device) - self.mat_lapboundary
          
    def forward(self, hs):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
                
        # src = voxel_features
        # pos = voxel_position_encoding
        # mask = voxel_features_padding_mask
        # #print(src.device, mask.device, self.query_embed.weight.device, pos.device)
        
        # if(curve_query_features is not None and primitive_type_embed is not None):
        #   assert(attention_mask is None)
        #   #add voxel embedding feature to voxel features
        #   voxel_features_mod = src + primitive_type_embed[0]
        #   curve_query_features_mod = curve_query_features + primitive_type_embed[2]
        #   src = torch.cat([curve_query_features_mod.transpose(0,1), voxel_features_mod], dim=0)
        #   #print(mask.shape, pos.shape)
        #   mask = F.pad(mask, (args.num_queries, 0), value=False)
        #   #TODO: add corner position encoding or query index position encoding
        #   pos = F.pad(pos, (0, 0, 0, 0, args.num_queries, 0), value=0)
          
        
        # #t0 = time.time()
        # assert mask is not None
        # hs = self.transformer(src, mask, self.query_embed.weight, pos, src_attention_mask=attention_mask)[0] #self.input_proj
        #t1 = time.time()
        #print("{}s elapsed for prepare data for transformer forwarding".format(t1-t0))
        #print("hs.shape=", hs.shape)[6, 2, 100, 192]
        
        #t0 = time.time()
        outputs_start_point_coord = self.patch_center_embed(hs).tanh()*0.5 # [-0.5,0.5]
        if not flag_using_scale:
          patch_shape_scale = 1#F.elu(self.patch_shape_scale_embed(hs)).unsqueeze(-2) + 1.0
        else:
          patch_shape_scale = F.elu(self.patch_shape_scale_embed(hs)).unsqueeze(-2) + 1.0
        
        parameterization_coord = torch.arange(points_per_patch_dim*points_per_patch_dim, dtype=torch.int32, device=hs.device)
        if args.enable_aux_loss:
        # if True:
          parameterization_coord = (torch.cat([(parameterization_coord // points_per_patch_dim).view(-1,1), (parameterization_coord % points_per_patch_dim).view(-1,1)], dim=1).float() / (points_per_patch_dim - 1)).view(1, 1, 1, points_per_patch_dim*points_per_patch_dim, 2).repeat(args.dec_layers, args.batch_size, args.num_patch_queries, 1, 1)
        else:
          parameterization_coord = (torch.cat([(parameterization_coord // points_per_patch_dim).view(-1,1), (parameterization_coord % points_per_patch_dim).view(-1,1)], dim=1).float() / (points_per_patch_dim - 1)).view(1, 1, 1, points_per_patch_dim*points_per_patch_dim, 2).repeat(1, args.batch_size, args.num_patch_queries, 1, 1)
        
        #output normals works only for no_pe
        if args.no_pe:
          #no pe
          sampled_points_feature = [torch.cat([self.patch_pe_x[i], self.patch_pe_y[j]], dim = -1) for i in range(points_per_patch_dim) for j in range(points_per_patch_dim)]
          sampled_points_feature = torch.cat(sampled_points_feature).view(1,1,1,points_per_patch_dim*points_per_patch_dim, 2).repeat(1, args.batch_size, args.num_patch_queries, 1,1)
          sampled_points = outputs_start_point_coord.unsqueeze(-2).repeat(1,1,1,points_per_patch_dim*points_per_patch_dim,1) + self.patch_shape_embed(sampled_points_feature, hs)[...,:3]
          if args.output_normal:
            sampled_normals = self.patch_shape_embed(sampled_points_feature, hs)[...,3:]
            #normalize normals
            sampled_normals = F.normalize(sampled_normals, dim=-1)
        elif args.spe:
          sampled_points_feature = [torch.cat([self.patch_pe_x[i], self.patch_pe_y[j]], dim = -1) for i in range(points_per_patch_dim) for j in range(points_per_patch_dim)]
          sampled_points_feature = torch.cat(sampled_points_feature).view(1,1,1,points_per_patch_dim*points_per_patch_dim, 6).repeat(1, args.batch_size, args.num_patch_queries, 1,1)
          sampled_points = outputs_start_point_coord.unsqueeze(-2).repeat(1,1,1,points_per_patch_dim*points_per_patch_dim,1) + self.patch_shape_embed(sampled_points_feature, hs)
        elif not args.pe_sin:
          if args.ori_mlp:
            sampled_points_feature = torch.cat([hs.unsqueeze(-2).repeat(1, 1, 1, points_per_patch_dim*points_per_patch_dim, 1), parameterization_coord], dim=-1)
            #in shape [6, bs, 100, 100*100, 3]
            sampled_points = outputs_start_point_coord.unsqueeze(-2).repeat(1,1,1,points_per_patch_dim*points_per_patch_dim,1) + patch_shape_scale*self.patch_shape_embed(sampled_points_feature)
          else:
            sampled_points_feature = [torch.cat([self.patch_pe_x.weight[i], self.patch_pe_y.weight[j]], dim = -1) for i in range(points_per_patch_dim) for j in range(points_per_patch_dim)]
            sampled_points_feature = torch.cat(sampled_points_feature).view(1,1,1,points_per_patch_dim*points_per_patch_dim, hn_pe_dim * 2).repeat(1, args.batch_size, args.num_patch_queries, 1,1)
            if flag_hn_with_coord:
              sampled_points_feature = torch.cat([sampled_points_feature, parameterization_coord], dim=-1)
            sampled_points = outputs_start_point_coord.unsqueeze(-2).repeat(1,1,1,points_per_patch_dim*points_per_patch_dim,1) + self.patch_shape_embed(sampled_points_feature, hs)
        else:
          sampled_points_feature = [torch.cat([self.patch_pe_x[i], self.patch_pe_y[j]], dim = -1) for i in range(points_per_patch_dim) for j in range(points_per_patch_dim)]
          sampled_points_feature = torch.cat(sampled_points_feature).view(1,1,1,points_per_patch_dim*points_per_patch_dim, hn_pe_dim * 2).repeat(1, args.batch_size, args.num_patch_queries, 1,1)
          sampled_points = outputs_start_point_coord.unsqueeze(-2).repeat(1,1,1,points_per_patch_dim*points_per_patch_dim,1) + self.patch_shape_embed(sampled_points_feature, hs)

        if args.patch_close:
          is_patch_closed_logits = self.closed_patch_embed(hs)
        is_patch_valid_pred = self.valid_patch_embed(hs)
        outputs_class = self.patch_type_prediction_embed(hs) #Circle, BSpline, Line, Ellipse or non-empty
        if not args.no_topo:
          if(args.normalize_embed_feature):
            output_patch_topo_embedding = F.normalize(self.patch_topo_embed(hs), dim=-1)
          else:
            if args.ori_topo:
              output_patch_topo_embedding = self.patch_topo_embed(hs)
            else:
              output_patch_topo_embedding_curve = self.patch_topo_embed_curve(hs)
              output_patch_topo_embedding_corner = self.patch_topo_embed_corner(hs)
        
        if not args.no_topo:
          if args.ori_topo:
            out = {'pred_patch_logits': is_patch_valid_pred[-1], 'pred_patch_type': outputs_class[-1], 'pred_patch_points': sampled_points[-1], 'patch_topo_embed': output_patch_topo_embedding[-1]}
          else:
            out = {'pred_patch_logits': is_patch_valid_pred[-1], 'pred_patch_type': outputs_class[-1], 'pred_patch_points': sampled_points[-1], 'patch_topo_embed_curve': output_patch_topo_embedding_curve[-1], 'patch_topo_embed_corner': output_patch_topo_embedding_corner[-1]}   
          if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(is_patch_valid_pred, outputs_class, sampled_points, output_patch_topo_embedding)
        else:
          out = {'pred_patch_logits': is_patch_valid_pred[-1], 'pred_patch_type': outputs_class[-1], 'pred_patch_points': sampled_points[-1]}  
        #t1 = time.time()
        #print("{}s elapsed for class and geometry embedding(auxilary included)".format(t1-t0))
        if args.patch_normal or args.output_normal:
          out['mask_x'] = self.mask_x
          out['mask_y'] = self.mask_y
        
        if args.patch_lap:
          out['mask_x_minus'] = self.mask_x_minus
          out['mask_x_plus'] = self.mask_x_plus
          out['mask_y_minus'] = self.mask_y_minus
          out['mask_y_plus'] = self.mask_y_plus
        
        if args.output_normal:
          out['pred_patch_normals'] = sampled_normals[-1]
        
        if args.patch_lapboundary:
          out['mat_lapboundary'] = self.mat_lapboundary
        
        if args.patch_close:
          out['closed_patch_logits'] = is_patch_closed_logits[-1]
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, is_patch_valid_pred, outputs_class, outputs_coord, output_patch_topo_embedding):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_patch_logits': e, 'pred_patch_type': a, 'pred_patch_points': b, "patch_topo_embed": d}
                for a, b, d, e in zip(outputs_class[:-1], outputs_coord[:-1], output_patch_topo_embedding[:-1], is_patch_valid_pred[:-1])]


class DETR_Shape_Tripath(nn.Module):
    def __init__(self, backbone, position_encoding, tripath_transformer, num_corner_queries, num_curve_queries, num_patch_queries, aux_loss=False, device = None):
        super().__init__()
        self.backbone = BackBone2VoxelTokens(backbone, position_encoding)
        self.tripath_transformer = tripath_transformer
        hidden_dim = tripath_transformer.d_model
        self.corner_model = DETR_Corner_Tripath(num_corner_queries, hidden_dim, aux_loss)
        self.curve_model = DETR_Curve_Tripath(num_curve_queries, hidden_dim, aux_loss, device)
        self.patch_model = DETR_Patch_Tripath(num_patch_queries, hidden_dim, aux_loss, device)
        if not args.no_tripath:
          if args.ori_tripath:
            self.primitive_type_embed = nn.Embedding(4, m*6) #voxel, corner, curve, patch
          else:
            self.primitive_type_embed = nn.Embedding(3, m*6) #voxel, corner, curve, patch
        else:
          self.primitive_type_embed = None
          
        
        #query embedding
        #corner
        self.query_embed_corner = nn.Embedding(num_corner_queries, hidden_dim)
        self.query_embed_curve = nn.Embedding(num_curve_queries, hidden_dim)
        self.query_embed_patch = nn.Embedding(num_patch_queries, hidden_dim)

    
    def forward(self, locations, features):
        if perform_profile:
          t0 = time.time()
        voxel_features, voxel_position_encoding, voxel_features_padding_mask, sparse_locations, attention_mask = self.backbone(locations, features)
        #locations: 15935,4
        #features: 15935,4
        #voxel_features: 300,1,192
        #voxel_position_encoding: 300,1,192
        #voxel_features_padding_mask: 1, 300
        # sparse_locations: tuple, 0,7,10
        #attention_mask: none
        if perform_profile:
          t1 = time.time()
          profile_dict['backbone_forwarding'].append(t1 - t0)

        
        #feature: pos, location and sparse location should be the same
        # corner_predictions, corner_query_features = self.corner_model(voxel_features, voxel_position_encoding, voxel_features_padding_mask, attention_mask)
        # #print("corner_query_features.shape =", corner_query_features.shape) #in shape [2, 100, 192]
        # #print("voxel_features.shape = ", voxel_features.shape, " voxel_position_encoding.shape", voxel_position_encoding.shape) [1598, 2, 192]
        # #print(self.primitive_type_embed.weight)
        # curve_predictions, curve_query_features = self.curve_model(voxel_features, voxel_position_encoding, voxel_features_padding_mask, attention_mask, corner_query_features=corner_query_features, primitive_type_embed=self.primitive_type_embed.weight)
        # patch_predictions                       = self.patch_model(voxel_features, voxel_position_encoding, voxel_features_padding_mask, attention_mask, curve_query_features=curve_query_features, primitive_type_embed=self.primitive_type_embed.weight)
        
       

        #transformer part
        src = voxel_features
        pos = voxel_position_encoding
        mask = voxel_features_padding_mask
        # print(src.device, mask.device, self.query_embed_corner.weight.device, pos.device)
        #t1 = time.time()
        #print("{}s elapsed for prepare data for transformer and encoding position".format(t1-t0))
        
        #t0 = time.time()
        #ori code
        if perform_profile:
          t0 = time.time()

        # print('src shape {} pos shape {} mask shape {}'.format(src.shape, pos.shape, mask.shape))
        assert mask is not None
        query_list = [self.query_embed_corner.weight, self.query_embed_curve.weight,self.query_embed_patch.weight]
        if not args.no_tripath:
          hs_corner, hs_curve, hs_patch = self.tripath_transformer(src, mask, query_list, pos, primitive_type_embed = self.primitive_type_embed.weight, src_attention_mask=attention_mask)[0] #self.input_proj, the first element
        else:
          hs_corner, hs_curve, hs_patch = self.tripath_transformer(src, mask, query_list, pos, primitive_type_embed = None, src_attention_mask=attention_mask)[0] #self.input_proj, the first element

        # print('hs corner: {} curve: {} patch: {}'.format(hs_corner.shape, hs_curve.shape, hs_patch.shape))
        #hs_corner: 6,1,100,192, 6 meanslayer
        if perform_profile:
          t1 = time.time()
          profile_dict['transformer_forwarding'].append(t1 - t0)


        #using transformer to get hs_corner, hs_curve, hs_patch
        if perform_profile:
          t0 = time.time()
        if args.vis_inter_layer == -1:
          corner_predictions = self.corner_model(hs_corner)
          curve_predictions = self.curve_model(hs_curve)
          patch_predictions = self.patch_model(hs_patch)
        else:
          corner_predictions = self.corner_model(hs_corner[args.vis_inter_layer].unsqueeze(0))
          curve_predictions = self.curve_model(hs_curve[args.vis_inter_layer].unsqueeze(0))
          patch_predictions = self.patch_model(hs_patch[args.vis_inter_layer].unsqueeze(0))

        if perform_profile:
          t1 = time.time()
          profile_dict['embedding_forwarding'].append(t1 - t0)

        return sparse_locations, corner_predictions, curve_predictions, patch_predictions

        

def output_prediction(filename, corner_predictions, curve_predictions, patch_predictions, corners_gt, curves_gt, patches_gt, corner_matching_indices, curve_matching_indices, patch_matching_indices, sample_id, sample_batch_idx=0):
  # print('save pkl begin')
  assert(len(corners_gt) == len(curves_gt) and len(curves_gt) == len(patches_gt))
  assert(sample_batch_idx < len(corners_gt))
  
  cur_corners_gt = corners_gt[sample_batch_idx]
  cur_curves_gt = curves_gt[sample_batch_idx]
  cur_patches_gt = patches_gt[sample_batch_idx]
  
  Result = {}
  
  Result['sample_id'] = sample_id
  
  #corners
  gt_corners_num = cur_corners_gt.shape[0]
  assert(cur_corners_gt.shape[1] == 3)
  Result['corners'] = {}
  Result['corners']['gt'] = cur_corners_gt.detach().cpu().numpy()
  Result['corners']['prediction'] = {}
  Result['corners']['prediction']['position'] = corner_predictions['pred_corner_position'][sample_batch_idx].detach().cpu().numpy()
  Result['corners']['prediction']['valid_prob'] = np.reshape(corner_predictions['pred_logits'].softmax(-1)[sample_batch_idx].detach().cpu().numpy(), [-1,2])[:,0]
  Result['corners']['corner_matching_indices'] = corner_matching_indices['indices'][sample_batch_idx]
  
  Result['corners']['corner_matching_indices'] = (Result['corners']['corner_matching_indices'][0].cpu().numpy(), Result['corners']['corner_matching_indices'][1].cpu().numpy())

  #curves  
  gt_curves_num = cur_curves_gt['curve_points'].shape[0]
  assert(len(cur_curves_gt['curve_points'].shape) == 3 and cur_curves_gt['curve_points'].shape[1] == 34 and cur_curves_gt['curve_points'].shape[2] == 3)
  Result['curves'] = {}
  Result['curves']['gt'] = {}
  Result['curves']['gt']['points'] = cur_curves_gt['curve_points'].cpu().numpy()
  Result['curves']['gt']['type'] = cur_curves_gt['labels'].cpu().numpy().astype(np.int32)
  Result['curves']['gt']['is_closed'] = cur_curves_gt["is_closed"].cpu().numpy().astype(np.int32)
  Result['curves']['prediction'] = {}
  Result['curves']['prediction']['points'] = curve_predictions['pred_curve_points'][sample_batch_idx].detach().cpu().numpy()
  Result['curves']['prediction']['valid_prob'] = np.reshape((curve_predictions['pred_curve_logits'].softmax(-1)[sample_batch_idx]).detach().cpu().numpy(), [-1,2])[:,0]
  Result['curves']['prediction']['closed_prob'] = np.reshape((curve_predictions['closed_curve_logits'].softmax(-1)[sample_batch_idx]).detach().cpu().numpy(), [-1,2])[:,1]
  Result['curves']['prediction']['type_prob'] = (curve_predictions['pred_curve_type'].softmax(-1)[sample_batch_idx]).detach().cpu().numpy()
  Result['curves']['curve_matching_indices'] = curve_matching_indices['indices'][sample_batch_idx]

  Result['curves']['curve_matching_indices'] = (Result['curves']['curve_matching_indices'][0].cpu().numpy(), Result['curves']['curve_matching_indices'][1].cpu().numpy())
  
  #patches
  Result['patches'] = {}
  Result['patches']['gt'] = {}
  Result['patches']['gt']['type'] = cur_patches_gt['labels'].cpu().numpy().astype(np.int32)
  if args.save_gt:
    Result['patches']['gt']['points'] = [item.cpu().numpy() for item in cur_patches_gt['patch_points']]
    
  
  Result['patches']['prediction'] = {}
  Result['patches']['prediction']['valid_prob'] = patch_predictions['pred_patch_logits'].softmax(-1)[sample_batch_idx].detach().cpu().numpy()[:,0]
  Result['patches']['prediction']['points'] = patch_predictions['pred_patch_points'][sample_batch_idx].detach().cpu().numpy()
  if args.output_normal:
    Result['patches']['prediction']['normals'] = patch_predictions['pred_patch_normals'][sample_batch_idx].detach().cpu().numpy()
  Result['patches']['prediction']['type_prob'] = patch_predictions['pred_patch_type'].softmax(-1)[sample_batch_idx].detach().cpu().numpy()
  
  if args.patch_close:
    Result['patches']['prediction']['closed_prob'] = np.reshape((patch_predictions['closed_patch_logits'].softmax(-1)[sample_batch_idx]).detach().cpu().numpy(), [-1,2])[:,1]
    

  Result['patches']['patch_matching_indices'] = patch_matching_indices['indices'][sample_batch_idx]

  Result['patches']['patch_matching_indices'] = (Result['patches']['patch_matching_indices'][0].cpu().numpy(), Result['patches']['patch_matching_indices'][1].cpu().numpy())

  if args.save_gt:
    Result['patch_curve_correspondence_gt'] = cur_patches_gt['patch_curve_correspondence'].cpu().numpy()
    #curve corner
    open_curve_idx = torch.where(cur_curves_gt['is_closed'] < 0.5)
    curve2corner_gt = cur_curves_gt['endpoints'][open_curve_idx] #selects open ones
    gt_cc = torch.zeros([cur_curves_gt['endpoints'].shape[0], gt_corners_num])
    gt_cc_open = torch.zeros([open_curve_idx[0].shape[0], gt_corners_num])
    
    gt_cc_open[torch.arange(curve2corner_gt.shape[0]), curve2corner_gt[:,0]] = 1
    gt_cc_open[torch.arange(curve2corner_gt.shape[0]), curve2corner_gt[:,1]] = 1
    gt_cc[open_curve_idx] = gt_cc_open
    # if gt_cc.shape[0] > 0 and gt_cc.shape[1] > 0:
    #   print('gt max: ', gt_cc.max())
    Result['curve_corner_correspondence_gt'] = gt_cc.numpy()
  #curve-corner topology
  if not args.no_topo:
    if args.ori_topo:
      corner_predictions_topo_embed = corner_predictions['corner_topo_embed'][sample_batch_idx] #in shape [100, 192]
      curve_predictions_topo_embed = curve_predictions['curve_topo_embed'][sample_batch_idx] #in shape[100, 192]
      patch_predictions_topo_embed = patch_predictions['patch_topo_embed'][sample_batch_idx] #in shape[100, 192]
      Result['curve_corner_similarity'] = torch.sigmoid(torch.mm(curve_predictions_topo_embed, corner_predictions_topo_embed.transpose(0,1))).detach().cpu().numpy() #in shape [100, 100]
      #patch-curve topology
      Result['patch_curve_similarity'] = torch.sigmoid(torch.mm(patch_predictions_topo_embed, curve_predictions_topo_embed.transpose(0,1))).detach().cpu().numpy() #in shape [100, 100]
    else:
      corner_predictions_topo_embed_curve = corner_predictions['corner_topo_embed_curve'][sample_batch_idx] #in shape [100, 192]
      curve_predictions_topo_embed_corner = curve_predictions['curve_topo_embed_corner'][sample_batch_idx] #in shape[100, 192]
      curve_predictions_topo_embed_patch = curve_predictions['curve_topo_embed_patch'][sample_batch_idx] #in shape[100, 192]
      patch_predictions_topo_embed_curve = patch_predictions['patch_topo_embed_curve'][sample_batch_idx] #in shape[100, 192]
      Result['curve_corner_similarity'] = torch.sigmoid(torch.mm(curve_predictions_topo_embed_corner, corner_predictions_topo_embed_curve.transpose(0,1))).detach().cpu().numpy() #in shape [100, 100]
      #patch-curve topology
      Result['patch_curve_similarity'] = torch.sigmoid(torch.mm(patch_predictions_topo_embed_curve, curve_predictions_topo_embed_patch.transpose(0,1))).detach().cpu().numpy() #in shape [100, 100]
      #patch corner
      corner_predictions_topo_embed_patch = corner_predictions['corner_topo_embed_patch'][sample_batch_idx] #in shape [100, 192]
      patch_predictions_topo_embed_corner = patch_predictions['patch_topo_embed_corner'][sample_batch_idx] #in shape[100, 192]
      Result['patch_corner_similarity'] = torch.sigmoid(torch.mm(patch_predictions_topo_embed_corner, corner_predictions_topo_embed_patch.transpose(0,1))).detach().cpu().numpy() #in shape [100, 100]
      
  # print('save pkl')
  with open(filename, "wb") as wf:
    pickle.dump(Result, wf)
  
  '''
  with open(filename, "w") as file:
    #first the gt data
    file.write("{} {}\n".format(gt_corners_num, gt_curves_num))
    for i in range(gt_corners_num):
      file.write("{} {} {}\n".format(cur_corners_gt[i][0], cur_corners_gt[i][1], cur_corners_gt[i][2]))
    assert(cur_curves_gt['labels'].shape[0] == cur_curves_gt['is_closed'].shape[0])
    for i in range(gt_curves_num):
      #type is_closed 
      file.write("{} {}".format(cur_curves_gt['labels'][i],  cur_curves_gt['is_closed'][i]))
   '''

def Curve_Corner_Matching_tripath(corner_predictions, curve_predictions, corners_gt, curves_gt, corner_indices, curve_indices, flag_round = False):
    #for samples in each batch seperately
    assert(len(corners_gt) == args.batch_size)
    assert(len(corners_gt) == len(curves_gt))
    
    topo_correspondence_loss = []
    topo_geometry_loss = []
    topo_correspondence_acc = []
    
    device = corner_predictions['pred_logits'].device
    
    zero_corners_examples = 0
    for i in range(len(corners_gt)):
      #no corners thus we do not have to compute
      # if(corners_gt[i].shape[0] == 0):
      if(corners_gt[i].shape[0] == 0 or corner_indices[i][0].shape[0]==0):
        #print("zero corners", i)
        zero_corners_examples += 1
        continue
      #compute pairwise dot product
      if args.ori_topo:
        corner_predictions_topo_embed = corner_predictions['corner_topo_embed'][i] #in shape [100, 192]
        curve_predictions_topo_embed = curve_predictions['curve_topo_embed'][i] #in shape[100, 192]
      else:
        corner_predictions_topo_embed = corner_predictions['corner_topo_embed_curve'][i] #in shape [100, 192]
        curve_predictions_topo_embed = curve_predictions['curve_topo_embed_corner'][i] #in shape[100, 192]
      
      #select matched curve and corners
      cur_corner_indices = corner_indices[i] #a tuple
      cur_curve_indices = curve_indices[i] #a tuple
      
      # corner_correspondence = torch.zeros_like(cur_corner_indices[0])
      # corner_correspondence[cur_corner_indices[1]] = cur_corner_indices[0]
      
      # curve_correspondence = torch.zeros_like(cur_curve_indices[0])
      # curve_correspondence[cur_curve_indices[1]] = cur_curve_indices[0]
      
      # valid_corner_predictions_topo_embed = corner_predictions_topo_embed[corner_correspondence]
      # valid_curve_predictions_topo_embed = curve_predictions_topo_embed[curve_correspondence]
      valid_corner_predictions_topo_embed = corner_predictions_topo_embed[cur_corner_indices[0]]
      valid_curve_predictions_topo_embed = curve_predictions_topo_embed[cur_curve_indices[0]]
      
      cur_curves_gt = curves_gt[i] #a dict
      open_curve_idx = torch.where(cur_curves_gt['is_closed'][cur_curve_indices[1]] < 0.5)
      
      if open_curve_idx[0].shape[0] == 0:
        zero_corners_examples += 1
        continue
      
      if(args.num_heads_dot > 1): #1
        curve_corner_similarity = torch.sigmoid(torch.einsum("ahf,bhf->abh", (valid_curve_predictions_topo_embed[open_curve_idx]).view(-1, args.num_heads_dot, args.topo_embed_dim//args.num_heads_dot), valid_corner_predictions_topo_embed.view(-1, args.num_heads_dot, args.topo_embed_dim//args.num_heads_dot)).max(-1).values)
      else:
        curve_corner_similarity = torch.sigmoid(torch.mm(valid_curve_predictions_topo_embed[open_curve_idx], valid_corner_predictions_topo_embed.transpose(0,1))) #in shape [valid_open_curves, valid_corners]
      #print(curve_corner_similarity)
      #print(curve_corner_similarity.shape)        
      
      curve2corner_gt = cur_curves_gt['endpoints'][cur_curve_indices[1]][open_curve_idx] #selects open ones
      # print(curve2corner_gt)
      '''
      print(curve2corner_gt)
      #write curves according to corner position and curve corner correspondences
      with open("gt_tmp_curve.obj", "w") as tmp_obj_file:
        for corner_position in corners_gt[i]:
          tmp_obj_file.write("v {} {} {}\n".format(corner_position[0], corner_position[1], corner_position[2]))
        for curve_endpoints in curve2corner_gt:
          tmp_obj_file.write("l {} {}\n".format(curve_endpoints[0] +1, curve_endpoints[1]+1))
      with open("pred_tmp_curve.obj", "w") as tmp_obj_file:
        for corner_position in corner_correspondence:
          tmp_obj_file.write("v {} {} {}\n".format(corner_predictions['pred_corner_position'][i][corner_position][0], corner_predictions['pred_corner_position'][i][corner_position][1], corner_predictions['pred_corner_position'][i][corner_position][2]))
        for curve_endpoints in curve2corner_gt:
          tmp_obj_file.write("l {} {}\n".format(curve_endpoints[0] +1, curve_endpoints[1]+1))
      input()
      '''
      assert(len(open_curve_idx) == 1)
      
      # max_corner = torch.zeros(1).type(torch.LongTensor)
      max_corner = 0
      if not curve2corner_gt.shape[0] == 0:
        # max_corner = torch.max(max_corner, curve2corner_gt.type(torch.LongTensor).max())
        max_corner = max(max_corner, curve2corner_gt.type(torch.LongTensor).max().item())
      if not cur_corner_indices[1].shape[0] == 0:
        # max_corner = torch.max(max_corner, cur_corner_indices[1].type(torch.LongTensor).max())
        max_corner = max(max_corner, cur_corner_indices[1].max().item())

      # max_corner = torch.max(curve2corner_gt.max(), cur_corner_indices[1].max())
      # gt_curve_corner_correspondence = torch.zeros([open_curve_idx[0].shape[0], max_corner.item() + 1])
      gt_curve_corner_correspondence = torch.zeros([open_curve_idx[0].shape[0], int(max_corner) + 1], device = corner_predictions['pred_logits'].device)

      gt_curve_corner_correspondence[torch.arange(curve2corner_gt.shape[0]), curve2corner_gt[:,0]] = 1
      gt_curve_corner_correspondence[torch.arange(curve2corner_gt.shape[0]), curve2corner_gt[:,1]] = 1
      gt_curve_corner_correspondence = gt_curve_corner_correspondence[:, cur_corner_indices[1]]
      assert(gt_curve_corner_correspondence.shape == curve_corner_similarity.shape)

      curve_endpoints_position = curve_predictions['pred_curve_points'][i][:,[0, -1]]#[100, 34, 3] -> [100, 2, 3]
      # valid_curve_endpoints_position = curve_endpoints_position[curve_correspondence]
      valid_curve_endpoints_position = curve_endpoints_position[cur_curve_indices[0]]

      #print(valid_curve_endpoints_position.shape)#[n_curves, 2, 3]
      
      # corner_position = corner_predictions['pred_corner_position'][i][corner_correspondence.to(device)] #!!!!!!
      # print ("idx0 shape", cur_corner_indices[0].shape)
      # print ("prediction shape: ", corner_predictions['pred_corner_position'][i].shape)
      corner_position = corner_predictions['pred_corner_position'][i][cur_corner_indices[0]] #to mapped space
      # print ("corner position", corner_position.shape)
      if cur_corner_indices[0].shape[0] != 0:
        # assert(len(corner_position.shape) == 2)
        # if not flag_eval:
        #   curve_corner_position = torch.cat([corner_position[curve2corner_gt[:,0]].unsqueeze(1), corner_position[curve2corner_gt[:,1]].unsqueeze(1)], axis=1)
        #   curve_corner_position_rev = torch.cat([corner_position[curve2corner_gt[:,1]].unsqueeze(1), corner_position[curve2corner_gt[:,0]].unsqueeze(1)], axis=1)
        # else:
        #   #there exists cases that the ground truth endpoints of the ground truth curve have no correspondence with the prediction points
        #   print ("idx shape", cur_corner_indices[1].shape)
        #   gt2valid_corner = -1 * torch.ones(cur_corner_indices[1].max() + 1, dtype = torch.long)
        #   gt2valid_corner[cur_corner_indices[1]] = torch.arange(cur_corner_indices[1].shape[0])
        #   print ('zero min: ', gt2valid_corner[curve2corner_gt[:,0]].min())
        #   print ('one min: ', gt2valid_corner[curve2corner_gt[:,1]].min())
        #   print ('curve2corner_gt: ', curve2corner_gt)
        #   print ('gt2valid ',gt2valid_corner)
        #   assert(gt2valid_corner[curve2corner_gt[:,0]].min()>=0)
        #   assert(gt2valid_corner[curve2corner_gt[:,1]].min()>=0)
        #   curve_corner_position = torch.cat([corner_position[gt2valid_corner[curve2corner_gt[:,0]]].unsqueeze(1), corner_position[gt2valid_corner[curve2corner_gt[:,1]]].unsqueeze(1)], axis=1)
        #   curve_corner_position_rev = torch.cat([corner_position[gt2valid_corner[curve2corner_gt[:,1]]].unsqueeze(1), corner_position[gt2valid_corner[curve2corner_gt[:,0]]].unsqueeze(1)], axis=1)
          
        # assert(len(curve_corner_position.shape) == 3 and curve_corner_position.shape[1] == 2)
        # diff_forward = (valid_curve_endpoints_position[open_curve_idx] - curve_corner_position).square().sum(-1).mean(-1).view(-1, 1)
        # diff_backward = (valid_curve_endpoints_position[open_curve_idx] - curve_corner_position_rev).square().sum(-1).mean(-1).view(-1, 1)
        # topo_geometry_loss.append(torch.cat([diff_forward, diff_backward], dim=1).min(-1).values.mean())

        # topo_geometry_loss.append(torch.zeros(1).to(corner_predictions['pred_logits'].device)[0])
        topo_geometry_loss.append(torch.zeros(1, device = corner_predictions['pred_logits'].device)[0])

        if not flag_round:
          # topo_correspondence_loss.append(F.binary_cross_entropy(curve_corner_similarity.view(-1), gt_curve_corner_correspondence.to(corner_predictions['pred_logits'].device).view(-1)))
          topo_correspondence_loss.append(F.binary_cross_entropy(curve_corner_similarity.view(-1), gt_curve_corner_correspondence.view(-1)))

        else:
          #rounding version of error
          # print('gt cc shape:{}\n'.format(gt_curve_corner_correspondence.shape), gt_curve_corner_correspondence)
          # print('pred cc\n', torch.round(curve_corner_similarity))

          
          # topo_correspondence_loss.append((torch.round(curve_corner_similarity)-gt_curve_corner_correspondence).abs().mean())
          
          #update 0107, divided by gt
          num_gt_curves = len(cur_curves_gt['is_closed'])
          num_gt_corners = len(corners_gt[i])
          # print('num of curves: {} corners: {}'.format(num_gt_curves, num_gt_corners))
          topo_correspondence_loss.append(((torch.round(curve_corner_similarity)-gt_curve_corner_correspondence).abs().sum() + num_gt_curves * num_gt_corners - gt_curve_corner_correspondence.shape[0] * gt_curve_corner_correspondence.shape[1] ) / (num_gt_curves * num_gt_corners))
          

          # summary_loss_dict['topo_patch_curve'] = ((pred_patch2curve - gt_patch2curve).abs().sum().item() + num_gt_patches * num_gt_curves - pred_patch2curve.shape[0] * pred_patch2curve.shape[1] ) / (num_gt_patches * num_gt_curves)
        
        if args.topo_acc:
          topo_correspondence_acc.append(100.0 * (1.0 - (torch.round(curve_corner_similarity)-gt_curve_corner_correspondence).abs().mean()) )
      
      if(False):
        #debug
        print("=============================================================")
        print("corner_matching_indices")
        print(cur_corner_indices)
        print("curve_matching_indices")
        print(cur_curve_indices)
        print("gt curve-corner correspondences")
        print(cur_curves_gt['endpoints'])
        print("gt closed curve label")
        print(open_curve_idx)
        print("gt matrix supervision")
        print(gt_curve_corner_correspondence)
        print(curve_corner_similarity.shape)
        print(gt_curve_corner_correspondence.shape)
        print(curve_corner_similarity)
        print("=============================================================")
      #print(curve_corner_similarity)
      #print(gt_curve_corner_correspondence)
    

    if not args.topo_acc:
      if(len(topo_geometry_loss) != 0):
        return sum(topo_geometry_loss) / len(topo_geometry_loss), sum(topo_correspondence_loss) / len(topo_correspondence_loss), zero_corners_examples == len(corners_gt)
      else:
        return torch.tensor(0, device=device), torch.tensor(0, device=device), zero_corners_examples == len(corners_gt)
    else:
      if(len(topo_geometry_loss) != 0):
        return sum(topo_geometry_loss) / len(topo_geometry_loss), sum(topo_correspondence_loss) / len(topo_correspondence_loss), zero_corners_examples == len(corners_gt), sum(topo_correspondence_acc) / len(topo_correspondence_acc)
      else:
        return torch.tensor(0, device=device), torch.tensor(0, device=device), zero_corners_examples == len(corners_gt), torch.tensor(100.0, device=device)

def Patch_Curve_Matching_tripath(curve_predictions, patch_predictions, curves_gt, patches_gt, curve_indices, patch_indices, flag_round = False):
    #for samples in each batch seperately
    assert(len(patches_gt) == args.batch_size)
    assert(len(patches_gt) == len(curves_gt))
    
    topo_correspondence_loss = []
    p2p_loss = []
    topo_correspondence_acc = []
    
    for i in range(len(patches_gt)):
      #no curves exists thus we do not have to compute
      if(curves_gt[i]['labels'].shape[0] == 0):
        continue
      #compute pairwise dot product
      if args.ori_topo:
        curve_predictions_topo_embed = curve_predictions['curve_topo_embed'][i] #in shape [100, 256]
        patch_predictions_topo_embed = patch_predictions['patch_topo_embed'][i] #in shape [100, 256]
      else:
        curve_predictions_topo_embed = curve_predictions['curve_topo_embed_patch'][i] #in shape [100, 256]
        patch_predictions_topo_embed = patch_predictions['patch_topo_embed_curve'][i] #in shape [100, 256]
      
      #select matched curve and corners
      cur_curve_indices = curve_indices[i] #a tuple
      cur_patch_indices = patch_indices[i] #a tuple
      if cur_curve_indices[0].shape[0] == 0:
        continue
      if cur_patch_indices[0].shape[0] == 0:
        continue

      # curve_correspondence = torch.zeros_like(cur_curve_indices[0])
      # curve_correspondence[cur_curve_indices[1]] = cur_curve_indices[0]
      
      # patch_correspondence = torch.zeros_like(cur_patch_indices[0])
      # patch_correspondence[cur_patch_indices[1]] = cur_patch_indices[0]
      
      # valid_patch_predictions_topo_embed = patch_predictions_topo_embed[patch_correspondence]
      # valid_curve_predictions_topo_embed = curve_predictions_topo_embed[curve_correspondence]
      
      valid_patch_predictions_topo_embed = patch_predictions_topo_embed[cur_patch_indices[0]]
      valid_curve_predictions_topo_embed = curve_predictions_topo_embed[cur_curve_indices[0]]
      
      if(args.num_heads_dot > 1):
        patch_curve_similarity = torch.sigmoid(torch.einsum("ahf,bhf->abh", (valid_patch_predictions_topo_embed).view(-1, args.num_heads_dot, args.topo_embed_dim//args.num_heads_dot), valid_curve_predictions_topo_embed.view(-1, args.num_heads_dot, args.topo_embed_dim//args.num_heads_dot)).max(-1).values)
      else:
        patch_curve_similarity = torch.sigmoid(torch.mm(valid_patch_predictions_topo_embed, valid_curve_predictions_topo_embed.transpose(0,1))) #in shape [valid_open_curves, valid_corners]
      #print(patch_curve_similarity)
      #print(patch_curve_similarity.shape)        
      # gt_patch_curve_correspondence = torch.from_numpy(patches_gt[i]['patch_curve_correspondence'].astype(np.float32))
      # gt_patch_curve_correspondence = torch.from_numpy(patches_gt[i]['patch_curve_correspondence'].astype(np.float32)).to(curve_predictions['pred_curve_logits'].device)[cur_patch_indices[1],][:,cur_curve_indices[1]]
      gt_patch_curve_correspondence = patches_gt[i]['patch_curve_correspondence'][cur_patch_indices[1],][:,cur_curve_indices[1]]

      #print(gt_patch_curve_correspondence.sum()) #should = 2*n_curves
      assert(gt_patch_curve_correspondence.shape == patch_curve_similarity.shape)
      if not flag_round:
        topo_correspondence_loss.append(F.binary_cross_entropy(patch_curve_similarity.view(-1), gt_patch_curve_correspondence.view(-1)))
      else:
        # topo_correspondence_loss.append((torch.round(patch_curve_similarity)-gt_patch_curve_correspondence).abs().mean())
        #modified 0107, divided by gt
        num_gt_curves = len(curves_gt[i]['labels'])
        num_gt_patches = len(patches_gt[i]['u_closed'])
        # print('num of curves: {} patches: {}'.format(num_gt_curves, num_gt_patches))

        topo_correspondence_loss.append(((torch.round(patch_curve_similarity)-gt_patch_curve_correspondence).abs().sum() + num_gt_curves * num_gt_patches - gt_patch_curve_correspondence.shape[0] * gt_patch_curve_correspondence.shape[1] ) / (num_gt_curves * num_gt_patches) )


        # topo_correspondence_loss.append(((torch.round(curve_corner_similarity)-gt_curve_corner_correspondence).abs().sum() + num_gt_curves * num_gt_corners - gt_curve_corner_correspondence.shape[0] * gt_curve_corner_correspondence.shape[1] ) / (num_gt_curves * num_gt_corners))
        
      if args.topo_acc:
        topo_correspondence_acc.append(100.0 * (1.0 - (torch.round(patch_curve_similarity)-gt_patch_curve_correspondence).abs().mean()) )
      
      if args.eval:
        patch_curve_similarity_round = torch.round(patch_curve_similarity)
        pred_p2p = torch.mm(patch_curve_similarity_round, torch.transpose(patch_curve_similarity_round, 0, 1))
        gt_p2p = torch.mm(gt_patch_curve_correspondence, torch.transpose(gt_patch_curve_correspondence, 0, 1))
        tmpid = torch.arange(patch_curve_similarity_round.shape[0], device=patch_curve_similarity_round.device)
        pred_p2p[tmpid, tmpid] = 0
        gt_p2p[tmpid, tmpid] = 0
        pred_p2p[pred_p2p > 0.5] = 1
        gt_p2p[gt_p2p > 0.5] = 1
        # p2p_loss.append((pred_p2p - gt_p2p).abs().mean().item())
        
        #modified 1223
        num_gt_patches = len(patches_gt[i]['u_closed'])
        num_match_patches = gt_p2p.shape[0]
        # print('gt patch: {} match patch: {}'.format(num_gt_patches, num_match_patches))
        p2p_loss.append( ((pred_p2p - gt_p2p).abs().sum().item() + num_gt_patches * num_gt_patches - num_match_patches * num_match_patches) / (num_gt_patches * num_gt_patches) )
      
      if(False):
        #debug
        print("=============================================================")
        print("corner_matching_indices")
        print(cur_corner_indices)
        print("curve_matching_indices")
        print(cur_curve_indices)
        print("gt curve-corner correspondences")
        print("gt closed curve label")
        print(open_curve_idx)
        print("gt matrix supervision")
        print(gt_curve_corner_correspondence)
        print(curve_corner_similarity.shape)
        print(gt_curve_corner_correspondence.shape)
        print(curve_corner_similarity)
        print("=============================================================")
    
    if args.eval:
      #return patch to patch mat
      if(len(topo_correspondence_loss) != 0):
        return sum(topo_correspondence_loss) / len(topo_correspondence_loss), sum(p2p_loss) / len(p2p_loss)
      else:
        return torch.tensor(0, device=curve_predictions['pred_curve_logits'].device), 0.0

    if not args.topo_acc:
      if(len(topo_correspondence_loss) != 0):
        return sum(topo_correspondence_loss) / len(topo_correspondence_loss)
      else:
        return torch.tensor(0, device=curve_predictions['pred_curve_logits'].device)
    else:
      if(len(topo_correspondence_loss) != 0):
        return sum(topo_correspondence_loss) / len(topo_correspondence_loss), sum(topo_correspondence_acc) / len(topo_correspondence_acc)
      else:
        return torch.tensor(0, device=curve_predictions['pred_curve_logits'].device), torch.tensor(100.0, device=curve_predictions['pred_curve_logits'].device)

def Patch_Corner_Matching_tripath(corner_predictions, curve_predictions, patch_predictions, corners_gt, curves_gt, patches_gt, corner_indices, curve_indices, patch_indices, flag_round = False): #rounding for evaluation
    #for samples in each batch seperately
    assert(len(patches_gt) == args.batch_size)
    assert(len(patches_gt) == len(curves_gt))
    
    topo_correspondence_loss = []
    topo_correspondence_acc = []
    #topo loss
    curve_point_loss = []
    curve_patch_loss = []
    patch_close_loss = []
    
    zero_corners_examples = 0
    for i in range(len(patches_gt)):
      #no curves exists thus we do not have to compute
      if(corners_gt[i].shape[0] == 0 or corner_indices[i][0].shape[0]==0):
        #no ground truth or no prediction
        #print("zero corners", i)
        zero_corners_examples += 1
        continue
      #compute pairwise dot product
      corner_predictions_topo_embed_patch = corner_predictions['corner_topo_embed_patch'][i] #in shape [100, 256]
      patch_predictions_topo_embed_corner = patch_predictions['patch_topo_embed_corner'][i] #in shape [100, 256]

      corner_predictions_topo_embed_curve = corner_predictions['corner_topo_embed_curve'][i] #in shape [100, 256]
      patch_predictions_topo_embed_curve = patch_predictions['patch_topo_embed_curve'][i] #in shape [100, 256]

      curve_predictions_topo_embed_corner = curve_predictions['curve_topo_embed_corner'][i] #in shape [100, 256]
      curve_predictions_topo_embed_patch = curve_predictions['curve_topo_embed_patch'][i] #in shape [100, 256]
      

      #select matched curve and corners
      cur_corner_indices = corner_indices[i] #a tuple
      cur_curve_indices = curve_indices[i] #a tuple
      cur_patch_indices = patch_indices[i] #a tuple

      # curve_correspondence = torch.zeros_like(cur_curve_indices[0])
      # curve_correspondence[cur_curve_indices[1]] = cur_curve_indices[0]
      
      # patch_correspondence = torch.zeros_like(cur_patch_indices[0])
      # patch_correspondence[cur_patch_indices[1]] = cur_patch_indices[0]

      # corner_correspondence = torch.zeros_like(cur_corner_indices[0])
      # corner_correspondence[cur_corner_indices[1]] = cur_corner_indices[0]
      
      # valid_patch_predictions_topo_embed_corner = patch_predictions_topo_embed_corner[patch_correspondence]
      # valid_corner_predictions_topo_embed_patch = corner_predictions_topo_embed_patch[corner_correspondence]
      
      # valid_patch_predictions_topo_embed_curve = patch_predictions_topo_embed_curve[patch_correspondence]
      # valid_corner_predictions_topo_embed_curve = corner_predictions_topo_embed_curve[corner_correspondence]

      # valid_curve_predictions_topo_embed_corner = curve_predictions_topo_embed_corner[curve_correspondence]
      # valid_curve_predictions_topo_embed_patch = curve_predictions_topo_embed_patch[curve_correspondence]

      valid_patch_predictions_topo_embed_corner = patch_predictions_topo_embed_corner[cur_patch_indices[0]]
      valid_corner_predictions_topo_embed_patch = corner_predictions_topo_embed_patch[cur_corner_indices[0]]
      
      valid_patch_predictions_topo_embed_curve = patch_predictions_topo_embed_curve[cur_patch_indices[0]]
      valid_corner_predictions_topo_embed_curve = corner_predictions_topo_embed_curve[cur_corner_indices[0]]

      valid_curve_predictions_topo_embed_corner = curve_predictions_topo_embed_corner[cur_curve_indices[0]]
      valid_curve_predictions_topo_embed_patch = curve_predictions_topo_embed_patch[cur_curve_indices[0]]

      #curve
      # open_curve_idx = torch.where(cur_curves_gt['is_closed'] < 0.5)
      cur_curves_gt = curves_gt[i] #a dict
      open_curve_idx = torch.where(cur_curves_gt['is_closed'][cur_curve_indices[1]] < 0.5)
      
      curve2corner_gt = cur_curves_gt['endpoints'][cur_curve_indices[1]][open_curve_idx]
      # print(open_curve_idx)
      # print('curve2corner', curve2corner_gt)
      assert(len(open_curve_idx) == 1)
      # gt_curve_corner_correspondence = torch.zeros([open_curve_idx[0].shape[0], corner_correspondence.shape[0]]) #target supervision
      # gt_curve_corner_correspondence[torch.arange(curve2corner_gt.shape[0]), curve2corner_gt[:,0]] = 1
      # gt_curve_corner_correspondence[torch.arange(curve2corner_gt.shape[0]), curve2corner_gt[:,1]] = 1

      # max_corner = torch.zeros(1)
      # if not curve2corner_gt.shape[0] == 0:
      #   max_corner = torch.max(max_corner, curve2corner_gt.max())
      # if not cur_corner_indices[1].shape[0] == 0:
      #   max_corner = torch.max(max_corner, cur_corner_indices[1].max())
      # # max_corner = torch.max(curve2corner_gt.max(), cur_corner_indices[1].max())
      # gt_curve_corner_correspondence = torch.zeros([open_curve_idx[0].shape[0], max_corner.type(torch.LongTensor).item() + 1])

      max_corner = 0
      if not curve2corner_gt.shape[0] == 0:
        # max_corner = torch.max(max_corner, curve2corner_gt.type(torch.LongTensor).max())
        max_corner = max(max_corner, curve2corner_gt.type(torch.LongTensor).max().item())
      if not cur_corner_indices[1].shape[0] == 0:
        # max_corner = torch.max(max_corner, cur_corner_indices[1].type(torch.LongTensor).max())
        max_corner = max(max_corner, cur_corner_indices[1].max().item())
      # max_corner = torch.max(curve2corner_gt.max(), cur_corner_indices[1].max())
      # gt_curve_corner_correspondence = torch.zeros([open_curve_idx[0].shape[0], max_corner.item() + 1])
      gt_curve_corner_correspondence = torch.zeros([open_curve_idx[0].shape[0], int(max_corner) + 1], device = corner_predictions['pred_logits'].device)
      # gt_curve_corner_correspondence = torch.zeros([open_curve_idx[0].shape[0], curve2corner_gt.max() + 1])
      gt_curve_corner_correspondence[torch.arange(curve2corner_gt.shape[0]), curve2corner_gt[:,0]] = 1
      gt_curve_corner_correspondence[torch.arange(curve2corner_gt.shape[0]), curve2corner_gt[:,1]] = 1
      gt_curve_corner_correspondence = gt_curve_corner_correspondence[:, cur_corner_indices[1]]
      # if(args.num_heads_dot > 1):
      #   patch_curve_similarity = torch.sigmoid(torch.einsum("ahf,bhf->abh", (valid_patch_predictions_topo_embed).view(-1, args.num_heads_dot, args.topo_embed_dim//args.num_heads_dot), valid_curve_predictions_topo_embed.view(-1, args.num_heads_dot, args.topo_embed_dim//args.num_heads_dot)).max(-1).values)
      # else:
      patch_corner_similarity = torch.sigmoid(torch.mm(valid_patch_predictions_topo_embed_corner, valid_corner_predictions_topo_embed_patch.transpose(0,1))) 
      curve_corner_similarity = torch.sigmoid(torch.mm(valid_curve_predictions_topo_embed_corner, valid_corner_predictions_topo_embed_curve.transpose(0,1)))

      patch_curve_similarity = torch.sigmoid(torch.mm(valid_patch_predictions_topo_embed_curve, valid_curve_predictions_topo_embed_patch.transpose(0,1)))

      #in shape [valid_open_curves, valid_corners]
      #print(patch_curve_similarity)
      #print(patch_curve_similarity.shape)        
      # gt_patch_curve_correspondence = torch.from_numpy(patches_gt[i]['patch_curve_correspondence'].astype(np.float32))
      # gt_patch_curve_correspondence = torch.from_numpy(patches_gt[i]['patch_curve_correspondence'].astype(np.float32)).to(corner_predictions['pred_logits'].device)[cur_patch_indices[1],][:,cur_curve_indices[1]]
      gt_patch_curve_correspondence = patches_gt[i]['patch_curve_correspondence'][cur_patch_indices[1],][:,cur_curve_indices[1]]


      gt_patch_corner_correspondence = torch.mm(gt_patch_curve_correspondence[:,open_curve_idx[0]], gt_curve_corner_correspondence)
      # tmp_mean = gt_patch_corner_correspondence[gt_patch_corner_correspondence > 1.0].mean()
      gt_patch_corner_correspondence[gt_patch_corner_correspondence > 1.0] = 1.0

      #print(gt_patch_curve_correspondence.sum()) #should = 2*n_curves
      assert(gt_patch_corner_correspondence.shape == patch_corner_similarity.shape)
      if not flag_round:
        # topo_correspondence_loss.append(F.binary_cross_entropy(patch_corner_similarity.view(-1), gt_patch_corner_correspondence.to(corner_predictions['pred_logits'].device).view(-1))) #binary cross entropy does not apply softmax
        topo_correspondence_loss.append(F.binary_cross_entropy(patch_corner_similarity.view(-1), gt_patch_corner_correspondence.view(-1))) #binary cross entropy does not apply softmax

      else:
        # topo_correspondence_loss.append((torch.round(patch_corner_similarity)-gt_patch_corner_correspondence).abs().mean())

        #modified 0107, divided by gt
        
        num_gt_curves = len(curves_gt[i]['is_closed'])
        num_gt_corners = len(corners_gt[i])
        num_gt_patches = len(patches_gt[i]['u_closed'])

        # print('len of corners: {} curves: {} patches: {}'.format(num_gt_corners, num_gt_curves, num_gt_patches))
        topo_correspondence_loss.append(((torch.round(patch_corner_similarity)-gt_patch_corner_correspondence).abs().sum() + num_gt_corners * num_gt_patches - gt_patch_corner_correspondence.shape[0] * gt_patch_corner_correspondence.shape[1] ) / (num_gt_corners * num_gt_patches))

        # topo_correspondence_loss.append(((torch.round(patch_curve_similarity)-gt_patch_curve_correspondence).abs().sum() + num_gt_curves * num_gt_patches - gt_patch_curve_correspondence.shape[0] * gt_patch_curve_correspondence.shape[1] ) / (num_gt_curves * num_gt_patches) )
      
      if args.topo_acc:
        topo_correspondence_acc.append(100.0 * (1.0 - (torch.round(patch_corner_similarity)-gt_patch_corner_correspondence).abs().mean()) )
        

      curve_point_loss.append((torch.sum(curve_corner_similarity[open_curve_idx[0]], dim=1) - 2).norm().mean()/math.sqrt(curve_corner_similarity.shape[1]))
      curve_patch_loss.append((torch.sum(patch_curve_similarity, dim=0) - 2).norm().mean()/math.sqrt(patch_curve_similarity.shape[0]))

      pc_cc_m = torch.mm(patch_curve_similarity[:, open_curve_idx[0]], curve_corner_similarity[open_curve_idx[0]])
      assert(pc_cc_m.shape == patch_corner_similarity.shape)
      patch_close_loss.append((pc_cc_m - 2 * patch_corner_similarity).norm() / math.sqrt(patch_corner_similarity.shape[0] * patch_corner_similarity.shape[1]))
      
      if(False):
        #debug
        print("=============================================================")
        print("corner_matching_indices")
        print(cur_corner_indices)
        print("curve_matching_indices")
        print(cur_curve_indices)
        print("gt curve-corner correspondences")
        print("gt closed curve label")
        print(open_curve_idx)
        print("gt matrix supervision")
        print(gt_curve_corner_correspondence)
        print(curve_corner_similarity.shape)
        print(gt_curve_corner_correspondence.shape)
        print(curve_corner_similarity)
        print("=============================================================")
    
    if not args.topo_acc:
      if(len(topo_correspondence_loss) != 0):
        return [sum(topo_correspondence_loss) / len(topo_correspondence_loss), sum(curve_point_loss) / len(curve_point_loss), sum(curve_patch_loss) / len(curve_patch_loss), sum(patch_close_loss) / len(patch_close_loss)]
      else:
        return [torch.tensor(0, device=corner_predictions['pred_logits'].device)] * 4
    else:
      if(len(topo_correspondence_loss) != 0):
        return [sum(topo_correspondence_loss) / len(topo_correspondence_loss), sum(curve_point_loss) / len(curve_point_loss), sum(curve_patch_loss) / len(curve_patch_loss), sum(patch_close_loss) / len(patch_close_loss), sum(topo_correspondence_acc) / len(topo_correspondence_acc)]
      else:
        return [torch.tensor(0, device=corner_predictions['pred_logits'].device)] * 4 + [torch.tensor(100.0, device=corner_predictions['pred_logits'].device)]


def Curve_Corner_Matching_v2(corner_predictions, curve_predictions, corners_gt, curves_gt, corner_indices, curve_indices):
    #for samples in each batch seperately
    assert(len(corners_gt) == args.batch_size)
    assert(len(corners_gt) == len(curves_gt))
    
    topo_correspondence_loss = []
    topo_geometry_loss = []
    
    device = corner_predictions['corner_topo_embed'].device
    
    zero_corners_examples = 0
    for i in range(len(corners_gt)):
      #no corners thus we do not have to compute
      if(corners_gt[i].shape[0] == 0):
        #print("zero corners", i)
        zero_corners_examples += 1
        continue
      #compute pairwise dot product
      corner_predictions_topo_embed = corner_predictions['corner_topo_embed'][i] #in shape [100, 192]
      curve_predictions_topo_embed = curve_predictions['curve_topo_embed'][i] #in shape[100, 192]
      
      #select matched curve and corners
      cur_corner_indices = corner_indices[i] #a tuple
      cur_curve_indices = curve_indices[i] #a tuple
      
      corner_correspondence = torch.zeros_like(cur_corner_indices[0])
      corner_correspondence[cur_corner_indices[1]] = cur_corner_indices[0]
      
      curve_correspondence = torch.zeros_like(cur_curve_indices[0])
      curve_correspondence[cur_curve_indices[1]] = cur_curve_indices[0]
      
      valid_corner_predictions_topo_embed = corner_predictions_topo_embed[corner_correspondence]
      valid_curve_predictions_topo_embed = curve_predictions_topo_embed[curve_correspondence]
      
      cur_curves_gt = curves_gt[i] #a dict
      open_curve_idx = torch.where(cur_curves_gt['is_closed'] < 0.5)
      
      if(args.num_heads_dot > 1): #1
        curve_corner_similarity = torch.sigmoid(torch.einsum("ahf,bhf->abh", (valid_curve_predictions_topo_embed[open_curve_idx]).view(-1, args.num_heads_dot, args.topo_embed_dim//args.num_heads_dot), valid_corner_predictions_topo_embed.view(-1, args.num_heads_dot, args.topo_embed_dim//args.num_heads_dot)).max(-1).values)
      else:
        curve_corner_similarity = torch.sigmoid(torch.mm(valid_curve_predictions_topo_embed[open_curve_idx], valid_corner_predictions_topo_embed.transpose(0,1))) #in shape [valid_open_curves, valid_corners]
      #print(curve_corner_similarity)
      #print(curve_corner_similarity.shape)        
      gt_curve_corner_correspondence = torch.zeros(curve_corner_similarity.shape) #target supervision
            
      curve2corner_gt = cur_curves_gt['endpoints'][open_curve_idx]
      '''
      print(curve2corner_gt)
      #write curves according to corner position and curve corner correspondences
      with open("gt_tmp_curve.obj", "w") as tmp_obj_file:
        for corner_position in corners_gt[i]:
          tmp_obj_file.write("v {} {} {}\n".format(corner_position[0], corner_position[1], corner_position[2]))
        for curve_endpoints in curve2corner_gt:
          tmp_obj_file.write("l {} {}\n".format(curve_endpoints[0] +1, curve_endpoints[1]+1))
      with open("pred_tmp_curve.obj", "w") as tmp_obj_file:
        for corner_position in corner_correspondence:
          tmp_obj_file.write("v {} {} {}\n".format(corner_predictions['pred_corner_position'][i][corner_position][0], corner_predictions['pred_corner_position'][i][corner_position][1], corner_predictions['pred_corner_position'][i][corner_position][2]))
        for curve_endpoints in curve2corner_gt:
          tmp_obj_file.write("l {} {}\n".format(curve_endpoints[0] +1, curve_endpoints[1]+1))
      input()
      '''
      assert(len(open_curve_idx) == 1)
      gt_curve_corner_correspondence[torch.arange(curve2corner_gt.shape[0]), curve2corner_gt[:,0]] = 1
      gt_curve_corner_correspondence[torch.arange(curve2corner_gt.shape[0]), curve2corner_gt[:,1]] = 1
      #print(gt_curve_corner_correspondence.sum()) #should = 2*n_curves
            
      curve_endpoints_position = curve_predictions['pred_curve_points'][i][:,[0, -1]]#[100, 34, 3] -> [100, 2, 3]
      valid_curve_endpoints_position = curve_endpoints_position[curve_correspondence]
      #print(valid_curve_endpoints_position.shape)#[n_curves, 2, 3]
      
      corner_position = corner_predictions['pred_corner_position'][i][corner_correspondence.to(device)] #!!!!!!
      assert(len(corner_position.shape) == 2)
      curve_corner_position = torch.cat([corner_position[curve2corner_gt[:,0]].unsqueeze(1), corner_position[curve2corner_gt[:,1]].unsqueeze(1)], axis=1)
      curve_corner_position_rev = torch.cat([corner_position[curve2corner_gt[:,1]].unsqueeze(1), corner_position[curve2corner_gt[:,0]].unsqueeze(1)], axis=1)
      assert(len(curve_corner_position.shape) == 3 and curve_corner_position.shape[1] == 2)
      diff_forward = (valid_curve_endpoints_position[open_curve_idx] - curve_corner_position).square().sum(-1).mean(-1).view(-1, 1)
      diff_backward = (valid_curve_endpoints_position[open_curve_idx] - curve_corner_position_rev).square().sum(-1).mean(-1).view(-1, 1)
      
      topo_geometry_loss.append(torch.cat([diff_forward, diff_backward], dim=1).min(-1).values.mean())      
      topo_correspondence_loss.append(F.binary_cross_entropy(curve_corner_similarity.view(-1), gt_curve_corner_correspondence.to(corner_predictions['corner_topo_embed'].device).view(-1)))
      
      if(False):
        #debug
        print("=============================================================")
        print("corner_matching_indices")
        print(cur_corner_indices)
        print("curve_matching_indices")
        print(cur_curve_indices)
        print("gt curve-corner correspondences")
        print(cur_curves_gt['endpoints'])
        print("gt closed curve label")
        print(open_curve_idx)
        print("gt matrix supervision")
        print(gt_curve_corner_correspondence)
        print(curve_corner_similarity.shape)
        print(gt_curve_corner_correspondence.shape)
        print(curve_corner_similarity)
        print("=============================================================")
      #print(curve_corner_similarity)
      #print(gt_curve_corner_correspondence)
    
    if(len(topo_geometry_loss) != 0):
      return sum(topo_geometry_loss) / len(topo_geometry_loss), sum(topo_correspondence_loss) / len(topo_correspondence_loss), zero_corners_examples == len(corners_gt)
    else:
      return torch.tensor(0, device=device), torch.tensor(0, device=device), zero_corners_examples == len(corners_gt)

def Patch_Curve_Matching(curve_predictions, patch_predictions, curves_gt, patches_gt, curve_indices, patch_indices):
    #for samples in each batch seperately
    assert(len(patches_gt) == args.batch_size)
    assert(len(patches_gt) == len(curves_gt))
    
    topo_correspondence_loss = []
    
    for i in range(len(patches_gt)):
      #no curves exists thus we do not have to compute
      if(curves_gt[i]['labels'].shape[0] == 0):
        continue
      #compute pairwise dot product
      curve_predictions_topo_embed = curve_predictions['curve_topo_embed'][i] #in shape [100, 192]
      patch_predictions_topo_embed = patch_predictions['patch_topo_embed'][i] #in shape [100, 192]
      
      #select matched curve and corners
      cur_curve_indices = curve_indices[i] #a tuple
      cur_patch_indices = patch_indices[i] #a tuple
           
      curve_correspondence = torch.zeros_like(cur_curve_indices[0])
      curve_correspondence[cur_curve_indices[1]] = cur_curve_indices[0]
      
      patch_correspondence = torch.zeros_like(cur_patch_indices[0])
      patch_correspondence[cur_patch_indices[1]] = cur_patch_indices[0]
      
      valid_patch_predictions_topo_embed = patch_predictions_topo_embed[patch_correspondence]
      valid_curve_predictions_topo_embed = curve_predictions_topo_embed[curve_correspondence]
            
      if(args.num_heads_dot > 1):
        patch_curve_similarity = torch.sigmoid(torch.einsum("ahf,bhf->abh", (valid_patch_predictions_topo_embed).view(-1, args.num_heads_dot, args.topo_embed_dim//args.num_heads_dot), valid_curve_predictions_topo_embed.view(-1, args.num_heads_dot, args.topo_embed_dim//args.num_heads_dot)).max(-1).values)
      else:
        patch_curve_similarity = torch.sigmoid(torch.mm(valid_patch_predictions_topo_embed, valid_curve_predictions_topo_embed.transpose(0,1))) #in shape [valid_open_curves, valid_corners]
      #print(patch_curve_similarity)
      #print(patch_curve_similarity.shape)        
      gt_patch_curve_correspondence = torch.from_numpy(patches_gt[i]['patch_curve_correspondence'].astype(np.float32))
      #print(gt_patch_curve_correspondence.sum()) #should = 2*n_curves
      assert(gt_patch_curve_correspondence.shape == patch_curve_similarity.shape)
      
      topo_correspondence_loss.append(F.binary_cross_entropy(patch_curve_similarity.view(-1), gt_patch_curve_correspondence.to(curve_predictions['curve_topo_embed'].device).view(-1)))
      
      if(False):
        #debug
        print("=============================================================")
        print("corner_matching_indices")
        print(cur_corner_indices)
        print("curve_matching_indices")
        print(cur_curve_indices)
        print("gt curve-corner correspondences")
        print("gt closed curve label")
        print(open_curve_idx)
        print("gt matrix supervision")
        print(gt_curve_corner_correspondence)
        print(curve_corner_similarity.shape)
        print(gt_curve_corner_correspondence.shape)
        print(curve_corner_similarity)
        print("=============================================================")
    
    if(len(topo_correspondence_loss) != 0):
      return sum(topo_correspondence_loss) / len(topo_correspondence_loss)
    else:
      return torch.tensor(0, device=curve_predictions['curve_topo_embed'].device)


def Curve_Corner_Matching(corner_predictions, curve_predictions, corners_gt, curves_gt, corner_indices, curve_indices):
    #return Curve_Corner_Matching_v2(corner_predictions, curve_predictions, corners_gt, curves_gt, corner_indices, curve_indices)
    #for samples in each batch seperately
    assert(len(corners_gt) == args.batch_size)
    assert(len(corners_gt) == len(curves_gt))
    
    topo_correspondence_loss = []
    topo_geometry_loss = []
    
    for i in range(len(corners_gt)):
      #no corners thus we do not have to compute
      if(corners_gt[i].shape[0] == 0):
        continue
      #compute pairwise dot product
      corner_predictions_topo_embed = corner_predictions['corner_topo_embed'][i] #in shape [100, 192]
      curve_predictions_topo_embed = curve_predictions['curve_topo_embed'][i] #in shape[100, 192]
      curve_corner_similarity = torch.sigmoid(torch.mm(curve_predictions_topo_embed, corner_predictions_topo_embed.transpose(0,1))) #in shape [100, 100]
      #print(curve_corner_similarity)
      #print(curve_corner_similarity.shape)
      gt_curve_corner_correspondence = torch.zeros([args.num_curve_queries, args.num_corner_queries]) #target supervision
      
      cur_curves_gt = curves_gt[i] #a dict
      cur_corner_indices = corner_indices[i] #a tuple
      cur_curve_indices = curve_indices[i] #a tuple
      
      corner_correspondence = torch.zeros_like(cur_corner_indices[0])
      corner_correspondence[cur_corner_indices[1]] = cur_corner_indices[0]
      
      curve_correspondence = torch.zeros_like(cur_curve_indices[0])
      curve_correspondence[cur_curve_indices[1]] = cur_curve_indices[0]
      
      '''
      print("corner")
      print(cur_corner_indices)
      print(corner_correspondence)
      
      print("curve")
      print(cur_curve_indices)
      print(curve_correspondence)
      print("==================================")
      '''
      
      open_curve_idx = torch.where(cur_curves_gt['is_closed'] < 0.5)
      curve2corner_gt = corner_correspondence[cur_curves_gt['endpoints'][open_curve_idx].view(-1)].view(-1, 2)
      open_curve_correspondence = curve_correspondence[open_curve_idx]
            
      gt_curve_corner_correspondence[open_curve_correspondence, curve2corner_gt[:,0]] = 1
      gt_curve_corner_correspondence[open_curve_correspondence, curve2corner_gt[:,1]] = 1
      #print(gt_curve_corner_correspondence.sum()) #should = 2*n_curves
      
      curve_endpoints_position = curve_predictions['pred_curve_points'][i][:,[0, -1]]#[100, 34, 3] -> [100, 2, 3]
      valid_curve_endpoints_position = curve_endpoints_position[open_curve_correspondence]
      #print(valid_curve_endpoints_position.shape)#[n_curves, 2, 3]
      
      valid_correspondence_mask = torch.zeros([args.num_curve_queries, args.num_corner_queries]) #target supervision
      valid_correspondence_mask[open_curve_correspondence] += 1
      valid_correspondence_mask[:, corner_correspondence] += 1
      curve_corner_similarity *= (valid_correspondence_mask > 1.5).float().to(device)
      gt_curve_corner_correspondence *= (valid_correspondence_mask > 1.5).float()
      
      
      corner_position = corner_predictions['pred_corner_position'][i]
      curve_corner_position = torch.cat([corner_position[curve2corner_gt[:,0]].unsqueeze(1), corner_position[curve2corner_gt[:,1]].unsqueeze(1)], axis=1)
      curve_corner_position_rev = torch.cat([corner_position[curve2corner_gt[:,1]].unsqueeze(1), corner_position[curve2corner_gt[:,0]].unsqueeze(1)], axis=1)
      diff_forward = (valid_curve_endpoints_position - curve_corner_position).square().sum(-1).mean(-1).view(-1, 1)
      diff_backward = (valid_curve_endpoints_position - curve_corner_position_rev).square().sum(-1).mean(-1).view(-1, 1)
      
      topo_geometry_loss.append(torch.cat([diff_forward, diff_backward], dim=1).min(-1).values.mean())
      topo_correspondence_loss.append(F.binary_cross_entropy(curve_corner_similarity.view(-1), gt_curve_corner_correspondence.to(device).view(-1), reduction='sum')/(len(open_curve_idx[0])*(corners_gt[i].shape[0])))
    
    if(len(topo_geometry_loss) != 0):
      return torch.tensor(topo_geometry_loss).mean(), torch.tensor(topo_correspondence_loss).mean()
    else:
      return torch.tensor([0], device=device), torch.tensor([0], device=device)

class SetCriterion_Corner(nn.Module):
    """ This class computes the loss for DETR-Corner.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth corner points and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and corner points)
    """
    def __init__(self, matcher, weight_dict, eos_coef, losses): #num_classes
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = 1#which is corner points
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef#used only for ce
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_corners, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        #target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes_o = torch.cat([torch.zeros(J.shape, device=src_logits.device) for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = 0 #target_classes_o, for corner points having matching gt, target label set to 0

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['corner_prediction_accuracy'] = accuracy(src_logits[idx], target_classes_o)[0]
            losses['corner_prediction_accuracy_overall'] = accuracy(src_logits.view(-1,2) , target_classes.view(-1))[0]
            
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_corners):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([v.shape[0] for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_geometry(self, outputs, targets, indices, num_corners):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_corner_position' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_corner_points = outputs['pred_corner_position'][idx]
        target_corner_points = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)

        #loss_geometry = F.mse_loss(src_corner_points, target_corner_points, reduction='none')
        if not args.geom_l2:
          loss_geometry = (src_corner_points - target_corner_points).square().sum(-1)
        else:
          loss_geometry = (src_corner_points - target_corner_points).norm(dim = -1)
        losses = {}
        losses['loss_geometry'] = loss_geometry.sum() / num_corners        
        return losses

    @torch.no_grad()
    def get_cd(self, outputs, targets, indices, num_corners):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        #length should be one
        assert 'pred_corner_position' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_corner_points = outputs['pred_corner_position'][idx]
        target_corner_points = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)

        #loss_geometry = F.mse_loss(src_corner_points, target_corner_points, reduction='none')
        loss_geometry = (src_corner_points - target_corner_points).square().sum(-1).sqrt()

        close_indices = torch.where(loss_geometry < args.dist_th)
        #valid elements
        losses = {}
        losses['cd'] = loss_geometry.sum() / max(1, loss_geometry.shape[0])     

        pred_logits = outputs['pred_logits'][0]
        device = pred_logits.device
        # tgt_lengths = torch.as_tensor([v.shape[0] for v in targets], device=device)
        # # Count the number of predictions that are NOT "no-object" (which is the last class)
        # card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        # card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        # losses = {'cardinality_error': card_err}
        pred_labels = pred_logits.softmax(-1)
        pred_valid_id = torch.where(pred_labels[:, 0]>args.val_th) #thresholding previously, this sentense might not be useful
        if num_corners == 1: #zero corners
          # losses['precision'] = torch.tensor(1.0, device=device)
          # losses['precision'] = torch.tensor(close_indices[0].shape[0] / max(1, pred_valid_id[0].shape[0]), device=device)
          if (pred_valid_id[0].shape[0] == 0):
            losses['precision'] = torch.tensor(1.0, device=device)
          else:
            losses['precision'] = torch.tensor(0.0, device=device)
          losses['recall'] = torch.tensor(1.0, device = device)
        else:
          losses['precision'] = torch.tensor(close_indices[0].shape[0] / max(1, pred_valid_id[0].shape[0]), device=device)
          losses['recall'] = torch.tensor(close_indices[0].shape[0] / num_corners, device = device)
        losses['fscore'] = 2 * losses['precision'] * losses['recall'] / (losses['precision'] + losses['recall'] + 1e-6)
        return losses
    # def get_precision(self, outputs, targets, indices, num_corners):

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_corners, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'geometry': self.loss_geometry,
            # 'precision': self.get_precision,
            # 'recall': self.get_recall,
            'cd': self.get_cd, #return precision and recall
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_corners, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        corner_matching_indices = {}
        corner_matching_indices['indices'] = indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_corners = sum(t.shape[0] for t in targets) 
        # num_corners = sum(id[0].shape[0] for id in indices)
        num_corners = torch.as_tensor([num_corners], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_corners)
        num_corners = torch.clamp(num_corners / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_corners))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            corner_matching_indices['aux_outputs'] = []
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                corner_matching_indices['aux_outputs'].append({'indices':indices})
                for loss in self.losses:
                    #if loss == 'masks':
                    #    # Intermediate masks losses are too costly to compute, we ignore them.
                    #    continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_corners, **kwargs)
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, corner_matching_indices

class SetCriterion_Curve(nn.Module):
    """ This class computes the loss for DETR-Curve.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth curve and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and curve geometry)
    """
    def __init__(self, matcher, weight_dict, eos_coef, losses): #num_classes
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = 4 #'Circle' 'BSpline' 'Line' 'Ellipse'
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(2) #non-empty, empty
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_closed_curve(self, outputs, targets, indices, num_curves, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "is_closed" containing a tensor of dim [nb_target_curves]
        prediction: closed_curve_logits
        """
        assert 'closed_curve_logits' in outputs
        
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['closed_curve_logits'][idx]
        target_classes = torch.cat([t["is_closed"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device)
        
        loss_curve_closed = F.cross_entropy(src_logits, target_classes)
        losses = {'loss_curve_closed': loss_curve_closed}
        if log:
            #all elements, only for current version
            losses['closed_accuracy_overall'] = accuracy(src_logits.view(-1,2), target_classes.view(-1))[0]
        
        return losses
    
    def loss_valid_labels(self, outputs, targets, indices, num_corners, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_curve_logits' in outputs
        src_logits = outputs['pred_curve_logits']

        idx = self._get_src_permutation_idx(indices)
        #target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes_o = torch.cat([torch.zeros(J.shape, device=src_logits.device) for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 1,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = 0 #target_classes_o, for corner points having matching gt, target label set to 0

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_valid_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['valid_class_accuracy'] = accuracy(src_logits[idx], target_classes_o)[0]
            losses['valid_class_accuracy_overall'] = accuracy(src_logits.view(-1,2), target_classes.view(-1))[0]
            # print('overall valid accuracy: ', losses['valid_class_accuracy_overall'])

        return losses
    
    def loss_type_labels(self, outputs, targets, indices, num_curves, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_curves]
        """
        assert 'pred_curve_type' in outputs
        src_logits = outputs['pred_curve_type']

        idx = self._get_src_permutation_idx(indices)
        # target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device)
        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        #target_classes_o = torch.cat([torch.zeros(J.shape, device=src_logits.device) for t, (_, J) in zip(targets, indices)])
        
        assert(len(src_logits[idx].shape) == 2 and src_logits[idx].shape[1] == 4)
        loss_ce = F.cross_entropy(src_logits[idx], target_classes)
        losses = {'loss_curve_type_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['type_class_accuracy'] = accuracy(src_logits[idx], target_classes)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_curves):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_curve_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_geometry(self, outputs, targets, indices, num_curves, cycleid = None):
        """Compute the losses related to the geometry, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_curve_points' in outputs
        idx = self._get_src_permutation_idx(indices) #only src
        src_curve_points = outputs['pred_curve_points'][idx]
        if args.flag_cycleid:
          # src_curve_cycleid = cycleid
          src_curve_cycleid = torch.cat(cycleid)
        # target_curve_points = torch.cat([t['curve_points'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_curve_points.device)
        target_curve_points = torch.cat([t['curve_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # target_curve_length_weight = torch.cat([t['curve_length_weighting'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_curve_points.device)
        target_curve_length_weight = torch.cat([t['curve_length_weighting'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        assert(target_curve_length_weight.shape[0] == target_curve_points.shape[0])
        is_target_curve_closed = torch.cat([t['is_closed'][i] for t, (_, i) in zip(targets, indices)])
        
        #print(src_curve_points.shape, target_curve_points.shape)
        assert(src_curve_points.shape == target_curve_points.shape)
        
        if(False):
          #compute chamfer distance
          pairwise_distance = torch.cdist(src_curve_points, target_curve_points, p=2.0) #in shape [batch_size, src_curve_number, tgt_curve_number)
          #print("pairwise_distance shape=", pairwise_distance.shape)
          s2t = pairwise_distance.min(-1).values.mean(-1)
          t2s = pairwise_distance.min(-2).values.mean(-1)        
          loss_geometry = (s2t + t2s) / 2.0
        else:
          #open curve
          if not args.flag_cycleid:
            if not args.geom_l2:
              #only_open works here
              distance_forward = (src_curve_points - target_curve_points).square().sum(-1).mean(-1).view(-1,1)
              distance_backward = (torch.flip(src_curve_points, dims=(1,)) - target_curve_points).square().sum(-1).mean(-1).view(-1,1)
              loss_geometry = torch.cat((distance_forward, distance_backward), dim=-1).min(-1).values
              #print(loss_geometry.shape)
              #print("src_curve_points.shape = ", src_curve_points.shape)
              if not args.curve_open_loss:
                for i in range(is_target_curve_closed.shape[0]):
                  if(is_target_curve_closed[i]):
                    tgt_possible_curves = cyclic_curve_points(target_curve_points[i].unsqueeze(0)) #[66, 34, 3]
                    loss_geometry[i] = (tgt_possible_curves - src_curve_points[i:i+1]).square().sum(-1).mean(-1).min()
            else:
              distance_forward = (src_curve_points - target_curve_points).norm(dim = -1).mean(-1).view(-1,1)
              distance_backward = (torch.flip(src_curve_points, dims=(1,)) - target_curve_points).norm(dim = -1).mean(-1).view(-1,1)
              loss_geometry = torch.cat((distance_forward, distance_backward), dim=-1).min(-1).values
              #print(loss_geometry.shape)
              #print("src_curve_points.shape = ", src_curve_points.shape)
              for i in range(is_target_curve_closed.shape[0]):
                if(is_target_curve_closed[i]):
                  tgt_possible_curves = cyclic_curve_points(target_curve_points[i].unsqueeze(0)) #[66, 34, 3]
                  loss_geometry[i] = (tgt_possible_curves - src_curve_points[i:i+1]).norm(dim = -1).mean(-1).min()
          else:

            src_curve_tmp = src_curve_points
            closeid = torch.where(is_target_curve_closed == True)[0]
            for i in range(closeid.shape[0]):
              src_curve_tmp[closeid[i]].roll(shifts=src_curve_cycleid[closeid[i]].item(), dims=0)
            rev_curve_tmp = torch.flip(src_curve_tmp, dims=(1,))
            
            if not args.geom_l2:
              distance_forward = (src_curve_tmp - target_curve_points).square().sum(-1).mean(-1).view(-1,1)
              distance_backward = (rev_curve_tmp - target_curve_points).square().sum(-1).mean(-1).view(-1,1)
            else:
              distance_forward = (src_curve_tmp - target_curve_points).norm(dim = -1).mean(-1).view(-1,1)
              distance_backward = (rev_curve_tmp - target_curve_points).norm(dim = -1).mean(-1).view(-1,1)
            loss_geometry = torch.cat((distance_forward, distance_backward), dim=-1).min(-1).values
          
        assert(loss_geometry.shape == target_curve_length_weight.shape)
        loss_geometry *= target_curve_length_weight
        losses = {}
        losses['loss_geometry'] = loss_geometry.sum() / num_curves        
        return losses

    def get_cd(self, outputs, targets, indices, num_curves):
        """Compute the losses related to the geometry, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_curve_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        pred_logits = outputs['pred_curve_logits'][0]
        device = pred_logits.device
        if indices[0][0].shape[0] != 0:
          src_curve_points = outputs['pred_curve_points'][idx]
          target_curve_points = torch.cat([t['curve_points'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_curve_points.device)
          target_curve_length_weight = torch.cat([t['curve_length_weighting'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_curve_points.device)
          assert(target_curve_length_weight.shape[0] == target_curve_points.shape[0])
          is_target_curve_closed = torch.cat([t['is_closed'][i] for t, (_, i) in zip(targets, indices)])
          
          #print(src_curve_points.shape, target_curve_points.shape)
          assert(src_curve_points.shape == target_curve_points.shape)
          
          if(True):
            #compute chamfer distance
            pairwise_distance = torch.cdist(src_curve_points, target_curve_points, p=2.0) #in shape [batch_size, src_curve_number, tgt_curve_number)
            #print("pairwise_distance shape=", pairwise_distance.shape)
            s2t = pairwise_distance.min(-1).values.mean(-1)
            t2s = pairwise_distance.min(-2).values.mean(-1)        
            loss_geometry = (s2t + t2s) / 2.0
          else:
            #open curve
            distance_forward = (src_curve_points - target_curve_points).square().sum(-1).mean(-1).view(-1,1).sqrt()
            distance_backward = (torch.flip(src_curve_points, dims=(1,)) - target_curve_points).square().sum(-1).mean(-1).view(-1,1).sqrt()
            loss_geometry = torch.cat((distance_forward, distance_backward), dim=-1).min(-1).values
            #print(loss_geometry.shape)
            #print("src_curve_points.shape = ", src_curve_points.shape)
            for i in range(is_target_curve_closed.shape[0]):
              if(is_target_curve_closed[i]):
                tgt_possible_curves = cyclic_curve_points(target_curve_points[i].unsqueeze(0)) #[66, 34, 3]
                loss_geometry[i] = (tgt_possible_curves - src_curve_points[i:i+1]).square().sum(-1).mean(-1).min()
          assert(loss_geometry.shape == target_curve_length_weight.shape)
          # loss_geometry *= target_curve_length_weight
          losses = {}
          losses['cd'] = loss_geometry.sum() / max(1, loss_geometry.shape[0])   

          #corner
          # losses['cd'] = loss_geometry.sum() / max(1, loss_geometry.shape[0])     
          close_indices = torch.where(loss_geometry < args.dist_th)
          
          # tgt_lengths = torch.as_tensor([v.shape[0] for v in targets], device=device)
          # # Count the number of predictions that are NOT "no-object" (which is the last class)
          # card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
          # card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
          # losses = {'cardinality_error': card_err}
          pred_labels = pred_logits.softmax(-1)
          pred_valid_id = torch.where(pred_labels[:, 0]>args.val_th)
          losses['precision'] = torch.tensor(close_indices[0].shape[0] / max(1, pred_valid_id[0].shape[0]), device=device)
          losses['recall'] = torch.tensor(close_indices[0].shape[0] / num_curves, device = device)
          
          #for classification
          src_logits = outputs['pred_curve_type']

          idx = self._get_src_permutation_idx(indices)
          target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device)
          #target_classes_o = torch.cat([torch.zeros(J.shape, device=src_logits.device) for t, (_, J) in zip(targets, indices)])
          losses['class_accuracy'] = accuracy(src_logits[idx], target_classes)[0]
        else:
          losses = {}
          losses['cd'] = torch.tensor(0.0, device = device)
          losses['precision'] = torch.tensor(0.0, device = device)
          losses['recall'] = torch.tensor(0.0, device = device)
          losses['class_accuracy'] = torch.tensor(0.0, device = device)
        
        losses['fscore'] = 2 * losses['precision'] * losses['recall'] / (losses['precision'] + losses['recall'] + 1e-6)

        return losses

    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_corners, **kwargs):
        loss_map = {
            'labels': self.loss_valid_labels,
            'curve_type': self.loss_type_labels,
            'cardinality': self.loss_cardinality,
            'geometry': self.loss_geometry,
            'closed_curve': self.loss_closed_curve,
            'cd': self.get_cd,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        if loss == 'geometry':
          return loss_map[loss](outputs, targets, indices, num_corners, **kwargs)
        else:
          return loss_map[loss](outputs, targets, indices, num_corners)
          

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        if not args.flag_cycleid:
          indices = self.matcher(outputs_without_aux, targets)
        else:
          indices, cycleid = self.matcher(outputs_without_aux, targets)
        curve_matching_indices = {}
        curve_matching_indices['indices'] = indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_corners = sum(len(t["labels"]) for t in targets)
        num_corners = torch.as_tensor([num_corners], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_corners)
        num_corners = torch.clamp(num_corners / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        if not args.flag_cycleid:
          for loss in self.losses:
              losses.update(self.get_loss(loss, outputs, targets, indices, num_corners))
        else:
          for loss in self.losses:
              losses.update(self.get_loss(loss, outputs, targets, indices, num_corners, cycleid = cycleid))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            curve_matching_indices['aux_outputs'] = []
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                curve_matching_indices['aux_outputs'].append({'indices':indices})
                for loss in self.losses:
                    #if loss == 'masks':
                    #    # Intermediate masks losses are too costly to compute, we ignore them.
                    #    continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_corners, **kwargs)
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, curve_matching_indices

class SetCriterion_Patch(nn.Module):
    """ This class computes the loss for DETR-Patch.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth patch and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and patch geometry)
    """
    def __init__(self, matcher, weight_dict, eos_coef, losses): #num_classes
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = 6 # Cylinder, Torus, BSpline, Plane, Cone, Sphere
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(2) #non-empty, empty
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        if args.patch_emd:
          self.emd_idlist = []
          base = torch.arange(points_per_patch_dim * points_per_patch_dim).view(points_per_patch_dim, points_per_patch_dim)
          # self.emd_idlist.append(base.flatten())
          for i in range(4):
            self.emd_idlist.append(torch.rot90(base, i, [0,1]).flatten())
          
          base_t = base.transpose(0,1)
          # self.emd_idlist.append(base_t.flatten())
          for i in range(4):
            self.emd_idlist.append(torch.rot90(base_t, i, [0,1]).flatten())

          self.emd_idlist = torch.cat(self.emd_idlist)

          #save emd list
          # np.savetxt('emdlist.txt', self.emd_idlist.view(-1,10).detach().numpy(),fmt = "%d")
        if args.patch_uv:
          self.emd_idlist_u = []
          self.emd_idlist_v = []
          base = torch.arange(points_per_patch_dim * points_per_patch_dim).view(points_per_patch_dim, points_per_patch_dim)
          #set idlist u
          for i in range(points_per_patch_dim):
            cur_base = base.roll(shifts=i, dims = 0)
            for i in range(0,4,2):
              self.emd_idlist_u.append(torch.rot90(cur_base, i, [0,1]).flatten())
            
            cur_base = cur_base.transpose(0,1)
            for i in range(1,4,2):
              self.emd_idlist_u.append(torch.rot90(cur_base, i, [0,1]).flatten())
          
          self.emd_idlist_u = torch.cat(self.emd_idlist_u)
          #set idlist v
          # for i in range(points_per_patch_dim):
          #   cur_base = base.roll(shifts=i, dims = 1)
          #   for i in range(4):
          #     self.emd_idlist_v.append(torch.rot90(cur_base, i, [0,1]).flatten())
            
          #   cur_base = cur_base.transpose(0,1)
          #   for i in range(4):
          #     self.emd_idlist_v.append(torch.rot90(cur_base, i, [0,1]).flatten())
          
          # self.emd_idlist_v = torch.cat(self.emd_idlist_v)

          # np.savetxt('emdlist_u.txt', self.emd_idlist_u.view(-1,10).detach().numpy(),fmt = "%d")
          # np.savetxt('emdlist_v.txt', self.emd_idlist_v.view(-1,10).detach().numpy(),fmt = "%d")

        if args.eval_param:
          cp_distance = ComputePrimitiveDistance(reduce = True)
          self.routines = {
            5: cp_distance.distance_from_sphere,
            0: cp_distance.distance_from_cylinder,
            4: cp_distance.distance_from_cone,
            3: cp_distance.distance_from_plane,
          }
          self.sqrt = True


    def loss_closed_patch(self, outputs, targets, indices, num_patches,log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "is_closed" containing a tensor of dim [nb_target_curves]
        prediction: closed_curve_logits
        """
        assert 'closed_patch_logits' in outputs
        
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['closed_patch_logits'][idx]
        target_classes = torch.cat([t["u_closed"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device)
        
        loss_curve_closed = F.cross_entropy(src_logits, target_classes)
        losses = {'loss_patch_closed': loss_curve_closed}

        if log:
            losses['closed_accuracy_overall'] = accuracy(src_logits.view(-1,2), target_classes.view(-1))[0]
        
        return losses
    
    def loss_valid_labels(self, outputs, targets, indices, num_patches, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_patch_logits' in outputs
        src_logits = outputs['pred_patch_logits']

        idx = self._get_src_permutation_idx(indices)
        #target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes_o = torch.cat([torch.zeros(J.shape, device=src_logits.device) for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 1,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = 0 #target_classes_o, for corner points having matching gt, target label set to 0

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_valid_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['valid_class_accuracy'] = accuracy(src_logits[idx], target_classes_o)[0]
            losses['valid_class_accuracy_overall'] = accuracy(src_logits.view(-1,2), target_classes.view(-1))[0]
        return losses
    
    def loss_type_labels(self, outputs, targets, indices, num_patches, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_curves]
        """
        assert 'pred_patch_type' in outputs
        src_logits = outputs['pred_patch_type']

        idx = self._get_src_permutation_idx(indices)
        # target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device)
        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        #target_classes_o = torch.cat([torch.zeros(J.shape, device=src_logits.device) for t, (_, J) in zip(targets, indices)])
        
        assert(len(src_logits[idx].shape) == 2 and src_logits[idx].shape[1] == 6)
        loss_ce = F.cross_entropy(src_logits[idx], target_classes)
        losses = {'loss_patch_type_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['type_class_accuracy'] = accuracy(src_logits[idx], target_classes)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_patches):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_patch_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_geometry(self, outputs, targets, indices, num_patches):
        """Compute the losses related to the geometry, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_patch_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_patch_points = outputs['pred_patch_points'][idx]
        if args.output_normal:
          src_patch_normals = outputs['pred_patch_normals'][idx]
        # print(len(targets))
        target_patch_points_list = [[t['patch_points'][j] for j in i.numpy().tolist()] for t, (_, i) in zip(targets, indices)]#torch.cat(, dim=0)
        # target_patch_area_weighting = torch.cat([t['patch_area_weighting'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_patch_points.device)
        target_patch_area_weighting = torch.cat([t['patch_area_weighting'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if args.patch_uv:
          target_patch_uclosed = torch.cat([t['u_closed'][i] for t, (_, i) in zip(targets, indices)], dim=0)
          target_patch_vclosed = torch.cat([t['v_closed'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        assert(target_patch_area_weighting.shape[0] == src_patch_points.shape[0])
        target_patch_points = target_patch_points_list[0]
        for i in range(1, len(target_patch_points_list)):
          target_patch_points += target_patch_points_list[i]

        if args.patch_normal or args.output_normal:
          target_patch_normals_list = [[t['patch_normals'][j] for j in i.numpy().tolist()] for t, (_, i) in zip(targets, indices)]#torch.cat(, dim=0)
          target_patch_normals = target_patch_normals_list[0]
          for i in range(1, len(target_patch_normals_list)):
            target_patch_normals += target_patch_normals_list[i]
        
        if args.patch_emd:
          #patch_normal and patch_lap not necessary here
          target_patch_points_batch = torch.cat(target_patch_points).view(len(target_patch_points), -1, 3)
          # loss_geometry_batch = (target_patch_points_batch - src_patch_points).sum(-1).mean()
          
          loss_geom = emd_by_id(target_patch_points_batch, src_patch_points, self.emd_idlist, points_per_patch_dim)
          
          if args.patch_uv:
            uclose_id = torch.where(target_patch_uclosed == 1)[0]
            # vclose_id = torch.where(target_patch_vclosed == 1)[0]
            #first v then u
            # if len(vclose_id) > 0:
            #   loss_geom[vclose_id] = emd_by_id(target_patch_points_batch[vclose_id], src_patch_points[vclose_id], self.emd_idlist_v, points_per_patch_dim)
            if len(uclose_id) > 0:
              loss_geom[uclose_id] = emd_by_id(target_patch_points_batch[uclose_id], src_patch_points[uclose_id], self.emd_idlist_u, points_per_patch_dim)
              

          losses = {}
          losses['loss_geometry'] = loss_geom.mean()
          return losses

        if args.batch_cd:
          #batch_computation
          #grid version begin
          # target_patch_points_batch = torch.cat(target_patch_points).view(len(target_patch_points), -1, 3)

          # #batch cd
          # target_nn = knn_points(target_patch_points_batch, src_patch_points)
          # target_cd = target_nn.dists[...,0] #N x 100
          # target_id = target_nn.idx #N x 100 x 1

          # if args.single_dir_patch_chamfer:
          #   loss_geometry_batch = ((target_cd.mean(-1)) * target_patch_area_weighting).mean()
          # else:
          #   src_nn = knn_points(src_patch_points, target_patch_points_batch)
          #   src_cd = src_nn.dists[...,0] 
          #   src_id = src_nn.idx[...,0] 
          #   # loss_geometry_batch = (target_cd.mean() * 1.0 + src_cd.mean() * 0.2) / 1.2
          #   loss_geometry_batch = ((target_cd.mean(-1)  + src_cd.mean(-1) * 0.2) * target_patch_area_weighting).mean() / 1.2
          #grid version end
          
          target_point_clouds_length = torch.tensor([len(p) for p in target_patch_points], device=src_patch_points.device)
          flag_equasize = False
          if len(target_point_clouds_length.unique()) == 1:
            flag_equasize = True
          
          target_patch_points_batch = list_to_padded(target_patch_points, (target_point_clouds_length.max(), 3), equisized = flag_equasize)

          target_nn = knn_points(target_patch_points_batch, src_patch_points, lengths1=target_point_clouds_length)
          target_cd = target_nn.dists[...,0]
          target_id = target_nn.idx
          if args.single_dir_patch_chamfer:
            loss_geometry_batch = ((target_cd.sum(-1) / target_point_clouds_length) * target_patch_area_weighting).mean()
          else:
            src_nn = knn_points(src_patch_points, target_patch_points_batch, lengths2=target_point_clouds_length)
            src_cd = src_nn.dists[...,0] 
            # src_id = src_nn.idx[...,0] 
            # loss_geometry_batch = (target_cd.mean() * 1.0 + src_cd.mean() * 0.2) / 1.2
            loss_geometry_batch = ((target_cd.sum(-1) / target_point_clouds_length  + src_cd.mean(-1) * 0.2) * target_patch_area_weighting).mean() / 1.2

          #emd:
          # loss_geometry_batch = (target_patch_points_batch - src_patch_points).sum(-1).mean()
          
          #laplacian loss
          loss_patch_lap = []
          if args.patch_lap:
            # for patch_idx in range(len(target_patch_points)):
            #   x_minus = src_patch_points[patch_idx][outputs['mask_x_minus']]
            #   x_plus = src_patch_points[patch_idx][outputs['mask_x_plus']]
            #   y_minus = src_patch_points[patch_idx][outputs['mask_y_minus']]
            #   y_plus = src_patch_points[patch_idx][outputs['mask_y_plus']]
            #   loss_patch_lap.append((src_patch_points[patch_idx] - (x_minus + x_plus + y_minus + y_plus) / 4.0).norm(dim = -1).mean())
          
            # loss_patch_lap_batch = sum(loss_patch_lap) / num_patches

            #batch version
            x_minus = src_patch_points[:,outputs['mask_x_minus'],:]
            x_plus = src_patch_points[:,outputs['mask_x_plus'],:]
            y_minus = src_patch_points[:,outputs['mask_y_minus'],:]
            y_plus = src_patch_points[:,outputs['mask_y_plus'],:]
            loss_patch_lap_batch = (src_patch_points - (x_minus + x_plus + y_minus + y_plus) / 4.0).norm(dim = -1).mean()
          
          if args.patch_normal:
            #single dir version
            # target_patch_normals_batch = torch.cat(target_patch_normals).view(len(target_patch_normals), -1, 3)

            target_patch_normals_batch = list_to_padded(target_patch_normals, (target_point_clouds_length.max(), 3), equisized = flag_equasize)

            if args.single_dir_patch_chamfer:
              #batch for df lengths ? remaining part should be zero
              # src_nn_pts = knn_gather(src_patch_points, target_id)
              # src_nn_pts = src_nn_pts.unsqueeze(-2)
              tangent_x = src_patch_points[:, outputs['mask_x'], ...] - src_patch_points
              tangent_y = src_patch_points[:, outputs['mask_y'], ...] - src_patch_points
              
              tangent_x_nn = knn_gather(tangent_x, target_id, target_point_clouds_length).squeeze(-2)
              tangent_y_nn = knn_gather(tangent_y, target_id, target_point_clouds_length).squeeze(-2)
              tangent_x_nn = F.normalize(tangent_x_nn, dim = -1)
              tangent_y_nn = F.normalize(tangent_y_nn, dim = -1)
              # loss_patch_normal_batch = (tangent_x_nn * target_patch_normals_batch).sum(-1).abs().mean() + (tangent_y_nn * target_patch_normals_batch).sum(-1).abs().mean()

              loss_patch_normal_batch = ((tangent_x_nn * target_patch_normals_batch).sum(-1).abs().sum(-1) / target_point_clouds_length + (tangent_y_nn * target_patch_normals_batch).sum(-1).abs().sum(-1) / target_point_clouds_length).mean()


          losses = {}
          losses['loss_geometry'] = loss_geometry_batch
          # if args.patch_normal:
          #   losses['loss_patch_normal'] = sum(loss_patch_normal) / num_patches   
          if args.patch_lap: #no patch_lapboundary
            losses['loss_patch_lap'] = loss_patch_lap_batch
          # if args.output_normal:
          #   losses['output_normal_diff'] = sum(loss_output_normal_diff) / num_patches
          #   losses['output_normal_tangent'] = sum(loss_output_normal_tangent) / num_patches

          if args.patch_normal:
            losses['loss_patch_normal'] = loss_patch_normal_batch
          return losses

        #print(len(src_patch_points), len(target_patch_points))
        assert(len(src_patch_points) == len(target_patch_points))
        assert(target_patch_area_weighting.shape[0] == len(target_patch_points))
        #compute chamfer distance
        loss_geometry = []
        loss_patch_normal = []
        loss_patch_lap = []
        loss_output_normal_diff = []
        loss_output_normal_tangent = []
        for patch_idx in range(len(target_patch_points)):
          if not args.geom_l2:
            patch_distance = torch.cdist(src_patch_points[patch_idx], target_patch_points[patch_idx], p=2.0).square() #in shape [src_patch_points, tgt_patch_points]
          else:
            patch_distance = torch.cdist(src_patch_points[patch_idx], target_patch_points[patch_idx], p=2.0) #in shape [src_patch_points, tgt_patch_points]
          assert(len(patch_distance.shape) == 2)
          if(args.single_dir_patch_chamfer):
            loss_geometry.append(target_patch_area_weighting[patch_idx]*patch_distance.min(0).values.mean())
          else:
            loss_geometry.append(target_patch_area_weighting[patch_idx]*(patch_distance.min(0).values.mean() + 0.2*patch_distance.min(-1).values.mean()) / 1.2)
            # loss_geometry.append((patch_distance.min(0).values.mean() + 0.2*patch_distance.min(-1).values.mean()) / 1.2)
          if args.patch_normal:
            if args.single_dir_patch_chamfer:
              closest_id = torch.argmin(patch_distance, dim = 0)
              # loss_patch_normal.append((src_patch_points[patch_idx] - target_patch_points[patch_idx][closest_id]).norm())

              # #for debugging
              # np.savetxt('init.xyz', src_patch_points[patch_idx].detach().cpu().numpy())
              # parameterization_coord = torch.arange(points_per_patch_dim*points_per_patch_dim, dtype=torch.int32, device=src_patch_points.device)
              # parameterization_coord = torch.cat([(parameterization_coord // points_per_patch_dim).view(-1,1), (parameterization_coord % points_per_patch_dim).view(-1,1)], dim=1).float().view(points_per_patch_dim*points_per_patch_dim, 2)
              # np.savetxt('xo.txt', parameterization_coord[:,0].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # np.savetxt('yo.txt', parameterization_coord[:,1].view(10,10).detach().cpu().numpy(), fmt = "%d")
              
              # tx = parameterization_coord[outputs['mask_x']] - parameterization_coord
              # ty = parameterization_coord[outputs['mask_y']] - parameterization_coord

              # np.savetxt('txx.txt', tx[:,0].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # np.savetxt('txy.txt', tx[:,1].view(10,10).det ach().cpu().numpy(), fmt = "%d")
              # np.savetxt('tyx.txt', ty[:,0].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # np.savetxt('tyy.txt', ty[:,1].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # np.savetxt('x.txt', parameterization_coord[:,0].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # np.savetxt('y.txt', parameterization_coord[:,1].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # np.savetxt('mx.txt', outputs['mask_x'].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # np.savetxt('my.txt', outputs['mask_y'].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # #id for mask is fine
              # #debugging end
              

              tangent_x = src_patch_points[patch_idx][outputs['mask_x']] - src_patch_points[patch_idx]
              tangent_y = src_patch_points[patch_idx][outputs['mask_y']] - src_patch_points[patch_idx]
              #normalize
              # tangent_x = tangent_x / tangent_x.norm(dim = -1).unsqueeze(dim=-1) 
              # tangent_y = tangent_y / tangent_y.norm(dim = -1).unsqueeze(dim=-1) 
              tangent_x = F.normalize(tangent_x, dim = -1)
              tangent_y = F.normalize(tangent_y, dim = -1)
              loss_patch_normal.append((tangent_x[closest_id] * target_patch_normals[patch_idx]).sum(-1).abs().mean())
              loss_patch_normal.append((tangent_y[closest_id] * target_patch_normals[patch_idx]).sum(-1).abs().mean())
            else:
              closest_id = torch.argmin(patch_distance, dim = 1)
              # loss_patch_normal.append((src_patch_points[patch_idx] - target_patch_points[patch_idx][closest_id]).norm())

              # #for debugging
              # np.savetxt('init.xyz', src_patch_points[patch_idx].detach().cpu().numpy())
              # parameterization_coord = torch.arange(points_per_patch_dim*points_per_patch_dim, dtype=torch.int32, device=src_patch_points.device)
              # parameterization_coord = torch.cat([(parameterization_coord // points_per_patch_dim).view(-1,1), (parameterization_coord % points_per_patch_dim).view(-1,1)], dim=1).float().view(points_per_patch_dim*points_per_patch_dim, 2)
              # np.savetxt('xo.txt', parameterization_coord[:,0].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # np.savetxt('yo.txt', parameterization_coord[:,1].view(10,10).detach().cpu().numpy(), fmt = "%d")
              
              # tx = parameterization_coord[outputs['mask_x']] - parameterization_coord
              # ty = parameterization_coord[outputs['mask_y']] - parameterization_coord

              # np.savetxt('txx.txt', tx[:,0].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # np.savetxt('txy.txt', tx[:,1].view(10,10).det ach().cpu().numpy(), fmt = "%d")
              # np.savetxt('tyx.txt', ty[:,0].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # np.savetxt('tyy.txt', ty[:,1].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # np.savetxt('x.txt', parameterization_coord[:,0].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # np.savetxt('y.txt', parameterization_coord[:,1].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # np.savetxt('mx.txt', outputs['mask_x'].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # np.savetxt('my.txt', outputs['mask_y'].view(10,10).detach().cpu().numpy(), fmt = "%d")
              # #id for mask is fine
              # #debugging end
              

              tangent_x = src_patch_points[patch_idx][outputs['mask_x']] - src_patch_points[patch_idx]
              tangent_y = src_patch_points[patch_idx][outputs['mask_y']] - src_patch_points[patch_idx]
              #normalize
              # tangent_x = tangent_x / tangent_x.norm(dim = -1).unsqueeze(dim=-1) 
              # tangent_y = tangent_y / tangent_y.norm(dim = -1).unsqueeze(dim=-1) 
              tangent_x = F.normalize(tangent_x, dim = -1)
              tangent_y = F.normalize(tangent_y, dim = -1)
              loss_patch_normal.append((tangent_x * target_patch_normals[patch_idx][closest_id]).sum(-1).abs().mean())
              loss_patch_normal.append((tangent_y * target_patch_normals[patch_idx][closest_id]).sum(-1).abs().mean())
          
          if args.output_normal:
            closest_id = torch.argmin(patch_distance, dim = 1)
            tangent_x = src_patch_points[patch_idx][outputs['mask_x']] - src_patch_points[patch_idx]
            tangent_y = src_patch_points[patch_idx][outputs['mask_y']] - src_patch_points[patch_idx]
            #normalize
            tangent_x = F.normalize(tangent_x, dim = -1)
            tangent_y = F.normalize(tangent_y, dim = -1)
            loss_output_normal_diff.append(torch.norm(target_patch_normals[patch_idx][closest_id] - src_patch_normals[patch_idx], dim = -1).mean())
            loss_output_normal_tangent.append((tangent_x * src_patch_normals[patch_idx]).sum(-1).abs().mean())
            loss_output_normal_tangent.append((tangent_y * src_patch_normals[patch_idx]).sum(-1).abs().mean())
          
          if args.patch_lap:
            x_minus = src_patch_points[patch_idx][outputs['mask_x_minus']]
            x_plus = src_patch_points[patch_idx][outputs['mask_x_plus']]
            y_minus = src_patch_points[patch_idx][outputs['mask_y_minus']]
            y_plus = src_patch_points[patch_idx][outputs['mask_y_plus']]
            loss_patch_lap.append((src_patch_points[patch_idx] - (x_minus + x_plus + y_minus + y_plus) / 4.0).norm(dim = -1).mean())
          if args.patch_lapboundary:
            loss_patch_lap.append(torch.mm(outputs['mat_lapboundary'], src_patch_points[patch_idx]).norm(dim = -1).mean())
            
        losses = {}
        losses['loss_geometry'] = sum(loss_geometry) / num_patches
        if args.patch_normal:
          losses['loss_patch_normal'] = sum(loss_patch_normal) / num_patches   
        if args.patch_lap or args.patch_lapboundary:
          losses['loss_patch_lap'] = sum(loss_patch_lap) / num_patches
        if args.output_normal:
          losses['output_normal_diff'] = sum(loss_output_normal_diff) / num_patches
          losses['output_normal_tangent'] = sum(loss_output_normal_tangent) / num_patches   
        return losses
    
    def loss_single_cd(self, outputs, targets, indices, num_patches):
        """Compute the losses related to the geometry, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_patch_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_patch_points = outputs['pred_patch_points'][idx]
        # print(len(targets))
        target_patch_points_list = [[t['patch_pcs'][j] for j in i.numpy().tolist()] for t, (_, i) in zip(targets, indices)]#torch.cat(, dim=0)
        target_patch_points = target_patch_points_list[0]
        for i in range(1, len(target_patch_points_list)):
          target_patch_points += target_patch_points_list[i]
        if True:
          #batch computation          
          target_point_clouds_length = torch.tensor([len(p) for p in target_patch_points], device=src_patch_points.device)
          flag_equasize = False
          if len(target_point_clouds_length.unique()) == 1:
            flag_equasize = True
          
          target_patch_points_batch = list_to_padded(target_patch_points, (target_point_clouds_length.max(), 3), equisized = flag_equasize)

          # print('target dtype: ', target_patch_points_batch.dtype)
          # print('src dtype: ',  src_patch_points.dtype)
          target_nn = knn_points(target_patch_points_batch, src_patch_points, lengths1=target_point_clouds_length)
          target_cd = target_nn.dists[...,0]
          loss_geometry_batch = (target_cd.sum(-1) / target_point_clouds_length).mean()
          losses = {}
          losses['loss_single_cd'] = loss_geometry_batch
          return losses
 
    def get_cd(self, outputs, targets, indices, num_patches):
        """Compute the losses related to the geometry, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_patch_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        # target_closeness = torch.cat([t["u_closed"][J] for t, (_, J) in zip(targets, indices)])
        # print('target closessness: ', target_closeness)

        src_patch_points = outputs['pred_patch_points'][idx]
        print('indices shape: ', indices[0][0].shape)
        #print(len(targets))
        target_patch_points_list = [[t['patch_points'][j] for j in i.numpy().tolist()] for t, (_, i) in zip(targets, indices)]#torch.cat(, dim=0)
        target_patch_area_weighting = torch.cat([t['patch_area_weighting'][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_patch_points.device)
        assert(target_patch_area_weighting.shape[0] == src_patch_points.shape[0])
        target_patch_points = target_patch_points_list[0]
        for i in range(1, len(target_patch_points_list)):
          target_patch_points += target_patch_points_list[i]
        
        #print(len(src_patch_points), len(target_patch_points))
        assert(len(src_patch_points) == len(target_patch_points))
        assert(target_patch_area_weighting.shape[0] == len(target_patch_points))
        #compute chamfer distance
        loss_geometry = []
        for patch_idx in range(len(target_patch_points)):
          patch_distance = torch.cdist(src_patch_points[patch_idx], target_patch_points[patch_idx].to(src_patch_points.device), p=2.0) #in shape [src_patch_points, tgt_patch_points]
          assert(len(patch_distance.shape) == 2)
          if(args.single_dir_patch_chamfer): #default: false
            loss_geometry.append(target_patch_area_weighting[patch_idx]*patch_distance.min(0).values.mean())
          else:
            # loss_geometry.append(target_patch_area_weighting[patch_idx]*(patch_distance.min(0).values.mean() + 0.2*patch_distance.min(-1).values.mean()) / 1.2)
            loss_geometry.append((patch_distance.min(0).values.mean() + patch_distance.min(-1).values.mean()) / 2.0)
        
        losses = {}
        losses['cd'] = sum(loss_geometry) / len(loss_geometry)

        if args.eval_res_cov:
          target_patch_pc_list = [[t['patch_pcs'][j] for j in i.numpy().tolist()] for t, (_, i) in zip(targets, indices)]#torch.cat(, dim=0)
          src_logits = outputs['closed_patch_logits'][idx]
          # print('src closessness: ', src_logits)
          src_uclosed = src_logits[:,0] < src_logits[:,1]
          target_classes = torch.cat([t["u_closed"][J] for t, (_, J) in zip(targets, indices)])
          # print('target closeness: ', target_classes)

          if args.eval_param:
            # print('idx: ', idx)
            src_with_param = outputs['pred_patch_with_param'][idx]
            src_type_logits = outputs['pred_patch_type'][idx]
            src_param = outputs['pred_patch_param'][idx]
            # print('src_param shape: ', src_param.shape)
            

          target_patch_pcs = target_patch_pc_list[0]
          for i in range(1, len(target_patch_pc_list)):
            target_patch_pcs += target_patch_pc_list[i]
          loss_res = []
          loss_res_filter = []
          patch_idx_filtered = []
          for patch_idx in range(len(target_patch_pcs)):
            # faces = get_patch_mesh_faces(args.points_per_patch_dim, args.points_per_patch_dim, src_uclosed[patch_idx])
            # mesh = trimesh.Trimesh(vertices = src_patch_points[patch_idx].detach().cpu().numpy(), faces = faces)

            if args.eval_param and src_with_param[patch_idx] > 0.5:
              #considering both distance to patch and parametric, because for cone, parametric distance is a little bit hight
              
              # para_dist = self.routines[torch.argmax(src_type_logits[patch_idx]).item()](src_patch_points[patch_idx], src_param[patch_idx], self.sqrt)
              para_dist = self.routines[torch.argmax(src_type_logits[patch_idx]).item()](target_patch_pcs[patch_idx], src_param[patch_idx], self.sqrt)
              # np.savetxt('plane.xyz', src_patch_points[patch_idx].detach().cpu().numpy())

              # if patch_idx == 0:
              #   np.savetxt('pycone_pred0.xyz', src_patch_points[patch_idx].detach().cpu().numpy())
              #   np.savetxt('pycone_gt0.xyz', target_patch_pcs[patch_idx].detach().cpu().numpy())
                

              # print('params: ',src_param[patch_idx])
              # print('patch type: ', torch.argmax(src_type_logits[patch_idx]).item())
              # print('patch param dist: {}'.format(para_dist))
              
              
              
              # loss_res.append(para_dist.item())
              # continue

            #spline version
            #extend version
            pts = src_patch_points[patch_idx].detach().cpu().numpy()
            # pts, faces = get_patch_mesh_pts_faces(pts, args.points_per_patch_dim, args.points_per_patch_dim, src_uclosed[patch_idx], True, 0.05)
            
            #prediction
            pts, faces = get_patch_mesh_pts_faces(pts, args.points_per_patch_dim, args.points_per_patch_dim, src_uclosed[patch_idx],0, True, 0.05)

            #ground truth
            # pts, faces = get_patch_mesh_pts_faces(target_patch_points[patch_idx].detach().cpu().numpy(), args.points_per_patch_dim, args.points_per_patch_dim, target_classes[patch_idx],0, True, 0.05)
            

            # print('pts size: ', pts.shape)
            # print('faces max', np.max(faces))
            mesh = trimesh.Trimesh(vertices = pts, faces = faces)


            # print("src patch closeness: ", src_uclosed[patch_idx])
            # mesh.export('patchmesh_{}.obj'.format(patch_idx))
            # np.savetxt('{}_patchpc.xyz'.format(patch_idx), target_patch_pcs[patch_idx].detach().cpu().numpy())

            # patch_distance = torch.cdist(src_patch_points[patch_idx], target_patch_pcs[patch_idx].to(src_patch_points.device), p=2.0) #in shape [src_patch_points, tgt_patch_points]
            # assert(len(patch_distance.shape) == 2)
            # loss_res.append(patch_distance.min(0).values.mean())
            
            #mesh version
            (closest_points,distances,triangle_id) = mesh.nearest.on_surface(target_patch_pcs[patch_idx].detach().cpu().numpy()) #here distance is squared norm
            # print('distance shape: ', distances.s
            # print('patch {} distance: {}'.format(patch_idx, distances.mean()))
            if args.eval_param and src_with_param[patch_idx] > 0.5:
              loss_res.append(min(para_dist.item(), distances.mean()))
            else:
              loss_res.append(distances.mean())

            if loss_res[-1] < args.th_res:
              loss_res_filter.append(loss_res[-1])
              patch_idx_filtered.append(indices[0][0][patch_idx].item())

          # print('patch idx before: ', patch_idx_filtered)

          #append unmatched id
          matched_id_set = set(indices[0][0].tolist())
          unmatched_id_set = set(range(outputs['pred_patch_points'][0].shape[0])) - matched_id_set
          patch_idx_filtered += list(unmatched_id_set)

          # print('patch idx after: ', patch_idx_filtered)
            
          # loss_res = torch.tensor(loss_res)
          losses['res'] = torch.tensor(sum(loss_res) / len(loss_res))
          if len(loss_res_filter) > 0:
            losses['res_filter'] = torch.tensor(sum(loss_res_filter) / len(loss_res_filter))
          else:
            losses['res_filter'] = torch.tensor(0.0)
          
          losses['n_patch'] = torch.tensor(len(loss_res))
          losses['n_patch_filter'] = torch.tensor(len(loss_res_filter))
          losses['patch_idx_filter'] = patch_idx_filtered


        loss_geometry = torch.tensor(loss_geometry)
        close_indices = torch.where(loss_geometry < args.dist_th)
        pred_logits = outputs['pred_patch_logits'][0]
        device = pred_logits.device
        # tgt_lengths = torch.as_tensor([v.shape[0] for v in targets], device=device)
        # # Count the number of predictions that are NOT "no-object" (which is the last class)
        # card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        # card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        # losses = {'cardinality_error': card_err}
        pred_labels = pred_logits.softmax(-1)
        pred_valid_id = torch.where(pred_labels[:, 0]>args.val_th)
        losses['precision'] = torch.tensor(close_indices[0].shape[0] / max(1, pred_valid_id[0].shape[0]), device=device)
        losses['recall'] = torch.tensor(close_indices[0].shape[0] / num_patches, device = device)
        losses['fscore'] = 2 * losses['precision'] * losses['recall'] / (losses['precision'] + losses['recall'] + 1e-6)

        assert 'pred_patch_type' in outputs
        src_logits = outputs['pred_patch_type']

        idx = self._get_src_permutation_idx(indices)
        # print('idx: ', idx)
        # print('close idx: ', close_indices)
        # print('idx shape {} close idx shape {}'.format(len(idx[0]), len(close_indices[0])))
        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device)
        #target_classes_o = torch.cat([torch.zeros(J.shape, device=src_logits.device) for t, (_, J) in zip(targets, indices)])

        losses['type_class_accuracy'] = accuracy(src_logits[idx], target_classes)[0]
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_patches, **kwargs):
        loss_map = {
            'labels': self.loss_valid_labels,
            'patch_type': self.loss_type_labels,
            'cardinality': self.loss_cardinality,
            'geometry': self.loss_geometry,
            'closed_patch': self.loss_closed_patch,
            'single_cd': self.loss_single_cd,
            'cd': self.get_cd,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_patches, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        #t0 = time.time()
        if(True):
          indices = self.matcher(outputs_without_aux, targets)
        else:
          chunk1 = {k: v[:2] for k, v in outputs_without_aux.items()}
          chunk2 = {k: v[2:].to('cuda:1') for k, v in outputs_without_aux.items()}
          indices = self.matcher(chunk1, targets[:2])
          indices2 = self.matcher(chunk2, targets[2:])
          indices += indices2
        #t1 = time.time()
        #print("patch assignment time cost:", t1-t0)
        
        patch_matching_indices = {}
        patch_matching_indices['indices'] = indices
        
        if len(indices) == 0:
          return {}, []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_corners = sum(len(t["labels"]) for t in targets)
        num_corners = torch.as_tensor([num_corners], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_corners)
        num_corners = torch.clamp(num_corners / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_corners))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            patch_matching_indices['aux_outputs'] = []
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                patch_matching_indices['aux_outputs'].append({'indices':indices})
                for loss in self.losses:
                    #if loss == 'masks':
                    #    # Intermediate masks losses are too costly to compute, we ignore them.
                    #    continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_corners, **kwargs)
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, patch_matching_indices


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def tf_summary_from_dict(loss_dict, is_training):  
  if(is_training):
    summary_name = "train_summary"
  else:
    summary_name = "test_summary"
  #create summary
  summary_list = []
  for item in loss_dict:
    summary_item = tf.compat.v1.Summary.Value(tag=summary_name + "/" + item, simple_value=loss_dict[item])
    summary_list.append(summary_item)
  return summary_list#tf.summary.merge(summary_list)

def save_on_master(*args, **kwargs):
    #if is_main_process():
    torch.save(*args, **kwargs)

def prepare_experiment_folders(exp_name):
  #prepare folders to write files
  if(not os.path.exists("experiments")): os.mkdir("experiments")
  experiment_dir = os.path.join("experiments", exp_name)
  if(not os.path.exists(experiment_dir)): os.mkdir(experiment_dir)
  log_dir = os.path.join(experiment_dir, "log")
  if(not os.path.exists(log_dir)): os.mkdir(log_dir)
  log_dir = os.path.join(log_dir, exp_name)
  if(not os.path.exists(log_dir)): os.mkdir(log_dir)

  obj_dir = os.path.join(experiment_dir, "obj")
  if(not os.path.exists(obj_dir)): os.mkdir(obj_dir)

  checkpoint_dir = os.path.join(experiment_dir, "ckpt")
  if(not os.path.exists(checkpoint_dir)): os.mkdir(checkpoint_dir)
  
  return log_dir, obj_dir, checkpoint_dir

def build_unified_model(device, flag_eval = False):
  ############# BackBone Network sparseCNN to Extract Features #############
  backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(Sparse_Backbone_Minkowski())
  
  ############# voxel_pos to position encoding #############
  position_encoding = PositionEmbeddingSine3D(out_voxel_dim, m*2, normalize=True)
  
  ############# build transformer #############
  corner_transformer = build_transformer(args)
  curve_transformer = build_transformer(args)
  patch_transformer = build_transformer(args)
  
  ############# build detr model #############
  model_shape = DETR_Shape(backbone, position_encoding, corner_transformer, curve_transformer, patch_transformer, args.num_queries, aux_loss=args.enable_aux_loss).to(device) #queries equals to 100, no aux loss
  
  ############# build matcher #############
  matcher_corner = build_matcher_corner(args, flag_eval = flag_eval)
  matcher_curve = build_matcher_curve(args, flag_eval = flag_eval)
  matcher_patch = build_matcher_patch(args, flag_eval = flag_eval)
  
  ############# build criterion #############
  # corner_weight_dict = {'loss_ce': args.class_loss_coef, 'loss_geometry': args.corner_geometry_loss_coef}
  # curve_weight_dict = {'loss_valid_ce': args.class_loss_coef, 'loss_geometry': args.curve_geometry_loss_coef, 'loss_curve_closed': 1, 'loss_curve_type_ce':args.class_loss_coef}
  # patch_weight_dict = {'loss_valid_ce': args.class_loss_coef, 'loss_geometry': args.patch_geometry_loss_coef, 'loss_patch_type_ce':args.class_loss_coef}

  if not flag_eval:
    corner_weight_dict = {'loss_ce': args.class_loss_coef, 'loss_geometry': args.corner_geometry_loss_coef}
    curve_weight_dict = {'loss_valid_ce': args.class_loss_coef, 'loss_geometry': args.curve_geometry_loss_coef, 'loss_curve_closed': 1, 'loss_curve_type_ce':args.class_loss_coef}
    patch_weight_dict = {'loss_valid_ce': args.class_loss_coef, 'loss_geometry': args.patch_geometry_loss_coef, 'loss_patch_type_ce':args.class_loss_coef}
    if args.enable_aux_loss:
      aux_weight_dict_corner = {}
      for i in range(args.dec_layers - 1):
        aux_weight_dict_corner.update({k + f'_aux_{i}': v for k, v in corner_weight_dict.items()})
      corner_weight_dict.update(aux_weight_dict_corner)
      
      aux_weight_dict_curve = {}
      for i in range(args.dec_layers - 1):
        aux_weight_dict_curve.update({k + f'_aux_{i}': v for k, v in curve_weight_dict.items()})
      curve_weight_dict.update(aux_weight_dict_curve)
    
    corner_losses = ['labels', 'cardinality', 'geometry']
    curve_losses = ['labels', 'cardinality', 'geometry', 'closed_curve', 'curve_type']
    patch_losses = ['labels', 'cardinality', 'geometry', 'patch_type']
  else:
    corner_weight_dict = {'loss_ce': args.class_loss_coef, 'loss_geometry': args.corner_geometry_loss_coef}
    curve_weight_dict = {'loss_valid_ce': args.class_loss_coef, 'loss_geometry': args.curve_geometry_loss_coef, 'loss_curve_closed': 1, 'loss_curve_type_ce':args.class_loss_coef}
    patch_weight_dict = {'loss_valid_ce': args.class_loss_coef, 'loss_geometry': args.patch_geometry_loss_coef, 'loss_patch_type_ce':args.class_loss_coef}
    # corner_losses = ['labels', 'cardinality', 'geometry']
    corner_losses = ['cd']
    # curve_losses = ['labels', 'cardinality', 'geometry', 'closed_curve', 'curve_type']
    curve_losses = ['cd']
    # patch_losses = ['labels', 'cardinality', 'geometry', 'patch_type']
    patch_losses = ['cd']

  if args.enable_aux_loss:
    aux_weight_dict_corner = {}
    for i in range(args.dec_layers - 1):
      aux_weight_dict_corner.update({k + f'_aux_{i}': v for k, v in corner_weight_dict.items()})
    corner_weight_dict.update(aux_weight_dict_corner)
    
    aux_weight_dict_curve = {}
    for i in range(args.dec_layers - 1):
      aux_weight_dict_curve.update({k + f'_aux_{i}': v for k, v in curve_weight_dict.items()})
    curve_weight_dict.update(aux_weight_dict_curve)
  
  # corner_losses = ['labels', 'cardinality', 'geometry']
  # curve_losses = ['labels', 'cardinality', 'geometry', 'closed_curve', 'curve_type']
  # patch_losses = ['labels', 'cardinality', 'geometry', 'patch_type']
  
  corner_loss_criterion = SetCriterion_Corner(matcher_corner, corner_weight_dict, corner_eos_coef_cal, corner_losses).to(device)
  curve_loss_criterion = SetCriterion_Curve(matcher_curve, curve_weight_dict, curve_eos_coef_cal, curve_losses).to(device)
  patch_loss_criterion = SetCriterion_Patch(matcher_patch, patch_weight_dict, patch_eos_coef_cal, patch_losses).to(device)
  
  ############# model statistics and optimizer #############
  n_parameters_backbone = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
  n_parameters_detr = sum(p.numel() for p in model_shape.parameters() if p.requires_grad)
  print('number of params:', n_parameters_backbone, n_parameters_detr)
  
  return model_shape, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion

class BackBoneWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, locations, features):
        locations_pos = locations[:,:3]
        locations_batch_idx = locations[:,-1:]
        locations = torch.cat([locations_batch_idx, locations_pos], dim=1)
        #t0 = time.time()
        output = self.backbone([locations, features])
        return output

class EncoderDecoder(nn.Module):
  def __init__(self, backbone, decoder, device = None):
    super().__init__()
    self.backbone = BackBoneWrapper(backbone)
    self.decoder = decoder
  def forward(self, locations, features):
    latent = self.backbone(locations, features)
    output = self.decoder(latent)
    # return output.C, output.F
    return output
  
def build_unified_model_tripath(device, flag_eval = False):
  ############# BackBone Network sparseCNN to Extract Features #############
  
  backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(Sparse_Backbone_Minkowski())
  ############# voxel_pos to position encoding #############
  position_encoding = PositionEmbeddingSine3D(out_voxel_dim, m*2, normalize=True)
  
  ############# build transformer #############
  # corner_transformer = build_transformer(args)
  # curve_transformer = build_transformer(args)
  # patch_transformer = build_transformer(args)

  #cur version
  tripath_transformer = build_transformer_tripath(args)
  # tripath_transformer = build_transformer_debug(args)
  # tripath_transformer = build_transformer(args)
  
  ############# build detr model #############
  # model_shape = DETR_Shape_Tripath(backbone, position_encoding, tripath_transformer, args.num_queries, aux_loss=args.enable_aux_loss).to(device) #queries equals to 100, no aux loss

  model_shape = DETR_Shape_Tripath(backbone, position_encoding, tripath_transformer, args.num_corner_queries, args.num_curve_queries, args.num_patch_queries, aux_loss=args.enable_aux_loss, device = device).to(device) #queries equals to 100, no aux loss

  # for debugging
  # print("model:", model_shape)

  
  ############# build matcher #############
  matcher_corner = build_matcher_corner(args, flag_eval = flag_eval)
  matcher_curve = build_matcher_curve(args, flag_eval = flag_eval)
  matcher_patch = build_matcher_patch(args, flag_eval = flag_eval)
  
  ############# build criterion #############
  if not flag_eval:
    corner_weight_dict = {'loss_ce': args.class_loss_coef, 'loss_geometry': args.corner_geometry_loss_coef}
    curve_weight_dict = {'loss_valid_ce': args.class_loss_coef, 'loss_geometry': args.curve_geometry_loss_coef, 'loss_curve_closed': 1, 'loss_curve_type_ce':args.class_loss_coef}

    patch_weight_dict = {'loss_valid_ce': args.class_loss_coef, 'loss_geometry': args.patch_geometry_loss_coef, 'loss_patch_type_ce':args.class_loss_coef}

    if args.patch_close:
      patch_weight_dict['loss_patch_closed'] = 1 #no weight yet

    if args.patch_normal:
      patch_weight_dict['loss_patch_normal'] = args.patch_normal_loss_coef
    
    if args.output_normal:
      patch_weight_dict['output_normal_diff'] = args.output_normal_diff_coef
      patch_weight_dict['output_normal_tangent'] = args.output_normal_tangent_coef
    
    if args.extra_single_chamfer:
      patch_weight_dict['loss_single_cd'] = args.extra_single_chamfer_weight
    
    if args.patch_lap or args.patch_lapboundary:
      # patch_weight_dict = {'loss_valid_ce': args.class_loss_coef, 'loss_geometry': args.patch_geometry_loss_coef, 'loss_patch_type_ce':args.class_loss_coef, 'loss_patch_lap': args.patch_lap_loss_coef}
      patch_weight_dict['loss_patch_lap'] = args.patch_lap_loss_coef
    if args.enable_aux_loss:
      aux_weight_dict_corner = {}
      for i in range(args.dec_layers - 1):
        aux_weight_dict_corner.update({k + f'_aux_{i}': v for k, v in corner_weight_dict.items()})
      corner_weight_dict.update(aux_weight_dict_corner)
      
      aux_weight_dict_curve = {}
      for i in range(args.dec_layers - 1):
        aux_weight_dict_curve.update({k + f'_aux_{i}': v for k, v in curve_weight_dict.items()})
      curve_weight_dict.update(aux_weight_dict_curve)
    
    corner_losses = ['labels', 'cardinality', 'geometry']
    curve_losses = ['labels', 'cardinality', 'geometry', 'closed_curve', 'curve_type']
    patch_losses = ['labels', 'cardinality', 'geometry', 'patch_type']
    if args.patch_close:
      patch_losses.append('closed_patch')
    if args.extra_single_chamfer:
      patch_losses.append('single_cd')
  else:
    corner_weight_dict = {'loss_ce': args.class_loss_coef, 'loss_geometry': args.corner_geometry_loss_coef}
    curve_weight_dict = {'loss_valid_ce': args.class_loss_coef, 'loss_geometry': args.curve_geometry_loss_coef, 'loss_curve_closed': 1, 'loss_curve_type_ce':args.class_loss_coef}
    patch_weight_dict = {'loss_valid_ce': args.class_loss_coef, 'loss_geometry': args.patch_geometry_loss_coef, 'loss_patch_type_ce':args.class_loss_coef}
    # corner_losses = ['labels', 'cardinality', 'geometry']
    corner_losses = ['cd']
    # curve_losses = ['labels', 'cardinality', 'geometry', 'closed_curve', 'curve_type']
    curve_losses = ['cd', 'closed_curve']
    # patch_losses = ['labels', 'cardinality', 'geometry', 'patch_type']
    patch_losses = ['cd', 'closed_patch']
  
  corner_loss_criterion = SetCriterion_Corner(matcher_corner, corner_weight_dict, corner_eos_coef_cal, corner_losses).to(device)
  curve_loss_criterion = SetCriterion_Curve(matcher_curve, curve_weight_dict, curve_eos_coef_cal, curve_losses).to(device)
  patch_loss_criterion = SetCriterion_Patch(matcher_patch, patch_weight_dict, patch_eos_coef_cal, patch_losses).to(device)
  
  ############# model statistics and optimizer #############
  n_parameters_backbone = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
  n_parameters_detr = sum(p.numel() for p in model_shape.parameters() if p.requires_grad)
  print('number of params:', n_parameters_backbone, n_parameters_detr)
  
  return model_shape, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion

flag_output_patch = True
patch_colormap = {'Plane': np.array([0,255,0]), 'Cylinder': np.array([255,0,0]),  'Torus': np.array([255,128,0]), 'BSpline': np.array([255,255,0]), 'Cone': np.array([255,102,255]), 'Sphere': np.array([0,0,255])}

def model_evaluation(model_shape, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion, train_data, device, train_iter, flag_output = True, test_folder = 'test_obj'):
  disable_aux_loss_output = True
  model_shape.eval()
  corner_loss_criterion.eval()
  curve_loss_criterion.eval()
  patch_loss_criterion.eval()
  assert(args.batch_size == 1)
  #test_data = train_data_loader(1, voxel_dim=voxel_dim, device=device, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, data_folder="test_data")
  data_loader_iterator = iter(train_data)
  
  obj_dir = os.path.join("experiments", args.experiment_name, test_folder)
  if(not os.path.exists(obj_dir)): os.mkdir(obj_dir)
  
  test_statistics = []
  sample_name_list = []
  
  sample_count = 0
  def export_curves(curve_points, points_number_per_curve, output_obj_file):
        curve_points = np.reshape(curve_points, [-1,3])
        with open(output_obj_file, "w") as wf:
          for point in curve_points:
            wf.write("v {} {} {}\n".format(point[0], point[1], point[2]))
          for i in range(curve_points.shape[0]):
            if(i % points_number_per_curve == (points_number_per_curve - 1)):
              continue
            wf.write("l {} {}\n".format(i+1, i+2))

  dict_sum = {}

  while(True):
    try:
      data_item = next(data_loader_iterator)
    except StopIteration:
      #data_loader_iterator = iter(train_data)
      #data_item = next(data_loader_iterator)
      break
    sample_count+=1
    locations = data_item[0].to(device)
    features = data_item[1].to(device)
    corner_points = data_item[2].to(device)
    corner_batch_idx = data_item[3].to(device)
    batch_sample_id = data_item[5]
    target_curves_list = data_item[6]
    target_patches_list = data_item[7]
    #convert target_curves_list and target_patches_list to cuda tensors
    for i in range(len(target_curves_list)):
      for k in target_curves_list[i]:
        target_curves_list[i][k] = target_curves_list[i][k].to(device)
    
    for i in range(len(target_patches_list)):
      for k in target_patches_list[i]:
        if k == 'patch_points' or k == 'patch_normals' or k == 'patch_pcs':
          for j in range(len(target_patches_list[i][k])): #list
            target_patches_list[i][k][j] = target_patches_list[i][k][j].to(device)
        else:
          target_patches_list[i][k] = target_patches_list[i][k].to(device)
    
    # if args.eval_res_cov:
    #   # input_pc = data_item[4]
    #   for i in range(len(target_patches_list)):
    #     k = 'patch_pc'
    #     for j in range(len(target_patches_list[i][k])): #list
    #       target_patches_list[i][k][j] = target_patches_list[i][k][j].to(device)

    if not args.patch_grid and not args.quicktest and not args.parsenet:
      # sample_id = int(batch_sample_id[0].replace(".pkl" ,""))
      sample_id = batch_sample_id[0].replace(".pkl" ,"")
    else:
      # sample_id = int(batch_sample_id[0].replace("_fix.pkl" ,""))
      sample_id = batch_sample_id[0].replace("_fix.pkl" ,"")
    print ('sample_id: ', sample_id)
    
    #supervision
    batch_corner_numbers = []
    for i in range(args.batch_size):
      batch_corner_numbers.append((corner_batch_idx==i).sum())
    target_corner_points_list = torch.split(corner_points, batch_corner_numbers)
    
    #forward
    sparse_locations, corner_predictions, curve_predictions, patch_predictions = model_shape(locations, features)#sparse locations not used here
    input_pointcloud = data_item[4][0]
    # continue
    
    #curves
    labels = torch.argmax(curve_predictions['pred_curve_logits'].softmax(-1)[0], dim=-1).cpu().numpy()
    #curve_type = torch.argmax(curve_predictions['pred_curve_type'].softmax(-1)[0], dim=-1).cpu().numpy()
    
    pred_curve_type = curve_type_list[torch.argmax(curve_predictions['pred_curve_type'].softmax(-1)[0], dim=-1).cpu().numpy().astype(np.int32)[np.where(labels == 0)]].tolist()
    gt_curve_type = curve_type_list[target_curves_list[0]['labels'].cpu().numpy().astype(np.int32)].tolist()
    if flag_output:
      with open(os.path.join(obj_dir, "{}_pred_curves_type.txt".format(sample_id)), "w") as f:
          f.write("{} {}\n".format(len(pred_curve_type), len(gt_curve_type)))
          for item in pred_curve_type: f.write("{}\n".format(item))
          f.write("==============================\n")
          for item in gt_curve_type: f.write("{}\n".format(item))

    if flag_output:
        # np.savetxt(os.path.join(obj_dir, "input_{:06d}.xyz".format(sample_id)), input_pointcloud)
        np.savetxt(os.path.join(obj_dir, "{}_0_input.xyz".format(sample_id)), input_pointcloud)
    
    if args.noise or args.partial:
      plywrite.write_ply(os.path.join(obj_dir, "{}_0_input.ply".format(sample_id)), input_pointcloud)
    
    curve_points = curve_predictions['pred_curve_points'][0].detach().cpu().numpy()
    effective_curve_points = np.reshape(curve_points[np.where(labels == 0)], [-1,3])
    #np.savetxt(os.path.join(obj_dir, "pred_curves_{:06d}.xyz".format(sample_id)), effective_curve_points)
    if flag_output:
      # export_curves(effective_curve_points, points_per_curve, os.path.join(obj_dir, "pred_curves_{:06d}.obj".format(sample_id)))
      curve_color = []
      # print('pred_curve_type shape:', len(pred_curve_type))
      for item in pred_curve_type:
        curve_color.append(np.expand_dims(curve_colormap[item],0))
      if len(curve_color) > 0:
        curve_color = np.concatenate(curve_color, 0)
        curve_color = np.tile(curve_color, points_per_curve).reshape([-1,3])
        # print('color shape: ', curve_color.shape)
        # print("point shape: ", effective_curve_points.shape)
        assert(curve_color.shape == effective_curve_points.shape)
        plywrite.save_vert_color_ply(effective_curve_points, curve_color, os.path.join(obj_dir, "{}_4_pred_curves.ply".format(sample_id)))
    
    empty_curve_points = np.reshape(curve_points[np.where(labels == 1)], [-1,3])
    #np.savetxt(os.path.join(obj_dir, "pred_curves_empty_{:06d}.xyz".format(sample_id)), empty_curve_points)
    
    #not output empty elements temporarily
    # if flag_output:
    #   export_curves(empty_curve_points, points_per_curve, os.path.join(obj_dir, "pred_curves_empty_{:06d}.obj".format(sample_id)))
    
    target_curve_points = np.reshape(target_curves_list[0]['curve_points'].cpu().numpy(), [-1,3])
    #np.savetxt(os.path.join(obj_dir, "gt_curves_{:06d}.xyz".format(sample_id)), target_curve_points)
    if flag_output:
      # export_curves(target_curve_points, points_per_curve, os.path.join(obj_dir, "gt_curves_{:06d}.obj".format(sample_id)))
      curve_color = []
      for item in gt_curve_type:
        curve_color.append(np.expand_dims(curve_colormap[item],0))
      curve_color = np.concatenate(curve_color, 0)
      curve_color = np.tile(curve_color, points_per_curve).reshape([-1,3])
      # print('color shape: ', curve_color.shape)
      # print("point shape: ", effective_curve_points.shape)
      assert(curve_color.shape == target_curve_points.shape)
      plywrite.save_vert_color_ply(target_curve_points, curve_color, os.path.join(obj_dir, "{}_3_gt_curves.ply".format(sample_id)))
    
    #corners
    labels = torch.argmax(corner_predictions['pred_logits'].softmax(-1)[0], dim=-1).cpu().numpy()
    corner_position = corner_predictions['pred_corner_position'][0].detach().cpu().numpy()
    effective_corner_position = corner_position[np.where(labels == 0)]
    if flag_output:
      np.savetxt(os.path.join(obj_dir, "{}_2_pred_corner.xyz".format(sample_id)), effective_corner_position)
    
    empty_corner_position = corner_position[np.where(labels == 1)]
    # empty corner not output
    # if flag_output:
    #   np.savetxt(os.path.join(obj_dir, "pred_corner_empty_{:06d}.xyz".format(sample_id)), empty_corner_position)
    
    target_corner_position = target_corner_points_list[0].cpu().numpy()
    if flag_output:
      np.savetxt(os.path.join(obj_dir, "{}_1_gt_corner.xyz".format(sample_id)), target_corner_position)
    
    #patches
    patch_labels = torch.argmax(patch_predictions['pred_patch_logits'].softmax(-1)[0], dim=-1).cpu().numpy()
    patch_points = patch_predictions['pred_patch_points'][0].detach().cpu().numpy() #in shape [100, 100*100, 3]
    effective_patch_points = patch_points[np.where(patch_labels == 0)]
    if flag_output:
    #  export_patches(effective_patch_points, os.path.join(obj_dir, "{:06d}_5_pred_patches.obj".format(sample_id)))
     export_patches_off(effective_patch_points, os.path.join(obj_dir, "{}_5_pred_patches.off".format(sample_id)))
     #save gt and pred patches
    
    pred_patch_type = patch_type_list[torch.argmax(patch_predictions['pred_patch_type'].softmax(-1)[0], dim=-1).cpu().numpy().astype(np.int32)[np.where(patch_labels == 0)]].tolist()
    gt_patch_type = patch_type_list[target_patches_list[0]['labels'].cpu().numpy().astype(np.int32)].tolist()
    if flag_output:
      with open(os.path.join(obj_dir, "{}_pred_patches_type.txt".format(sample_id)), "w") as f:
        f.write("{} {}\n".format(len(pred_patch_type), len(gt_patch_type)))
        for item in pred_patch_type: f.write("{}\n".format(item))
        f.write("==============================\n")
        for item in gt_patch_type: f.write("{}\n".format(item))
        
        if flag_output_patch:
          #gt and pred
          patch_color = []
          # print('len patch type: ', len(pred_patch_type))
          for item in pred_patch_type:
            patch_color.append(np.expand_dims(patch_colormap[item],0))
          
          if len(patch_color) > 0:
            patch_color = np.concatenate(patch_color, 0)
            patch_color = np.tile(patch_color, points_per_patch_dim * points_per_patch_dim).reshape([-1,3])
            # print('color shape: ', curve_color.shape)
            # print("point shape: ", effective_curve_points.shape)
            # assert(curve_color.shape == target_curve_points.shape)
            # print('patch points shape: ', effective_patch_points.shape)
            plywrite.save_vert_color_ply(effective_patch_points.reshape(-1,3), patch_color, os.path.join(obj_dir, "{}_7_pred_patchtype.ply".format(sample_id)))
          else:
            print('no pred patches')

          patch_color = []
          gt_patch_pts = target_patches_list[0]['patch_points']
          assert(len(gt_patch_pts) == len(gt_patch_type))
          gt_patch_pts_all = []
          for i in range(len(gt_patch_pts)):
            # np.savetxt("{}_patch.xyz".format(i), gt_patch_pts[i].cpu().numpy())
            gt_patch_pts_all.append(gt_patch_pts[i].cpu().numpy())
            item = gt_patch_type[i]
            for j in range(gt_patch_pts[i].shape[0]):
              patch_color.append(np.expand_dims(patch_colormap[item],0))

          gt_patch_pts_all = np.concatenate(gt_patch_pts_all, 0)  
          patch_color = np.concatenate(patch_color, 0)
          plywrite.save_vert_color_ply(gt_patch_pts_all, patch_color, os.path.join(obj_dir, "{}_6_gt_patchtype.ply".format(sample_id)))

    max_norm = 0
    #losses
    curve_loss_dict, curve_matching_indices = curve_loss_criterion(curve_predictions, target_curves_list)
    curve_weight_dict = curve_loss_criterion.weight_dict
    corner_loss_dict, corner_matching_indices = corner_loss_criterion(corner_predictions, target_corner_points_list)
    corner_weight_dict = corner_loss_criterion.weight_dict
    patch_loss_dict, patch_matching_indices = patch_loss_criterion(patch_predictions, target_patches_list)
    patch_weight_dict = patch_loss_criterion.weight_dict
    

    
    #print(loss_dict)
    # corner_losses = sum(corner_loss_dict[k] * corner_weight_dict[k] for k in corner_loss_dict.keys() if k in corner_weight_dict)
    # curve_losses = sum(curve_loss_dict[k] * curve_weight_dict[k] for k in curve_loss_dict.keys() if k in curve_weight_dict)
    # patch_losses = sum(patch_loss_dict[k] * patch_weight_dict[k] for k in patch_loss_dict.keys() if k in patch_weight_dict)
    # losses = corner_losses + curve_losses + patch_losses
        
    summary_loss_dict = {}
    if len(patch_matching_indices) == 0:
      continue
    for k in corner_loss_dict.keys():
      # if k in corner_weight_dict:
      #   if(disable_aux_loss_output and "_aux_" in k):
      #     continue
      #   summary_loss_dict["corner_" + k] = corner_weight_dict[k] * corner_loss_dict[k].cpu().detach().numpy()
      summary_loss_dict["corner_" + k] = corner_loss_dict[k].cpu().detach().numpy()
    
    for k in curve_loss_dict.keys():
      # if k in curve_weight_dict:
      #   if(disable_aux_loss_output and "_aux_" in k):
      #     continue
      #   summary_loss_dict["curve_" + k] = curve_weight_dict[k] * curve_loss_dict[k].cpu().detach().numpy()
      summary_loss_dict["curve_" + k] = curve_loss_dict[k].cpu().detach().numpy()
    
    for k in patch_loss_dict.keys():
      # if k in patch_weight_dict:
      #   if(disable_aux_loss_output and "_aux_" in k):
      #     continue
      #   summary_loss_dict["patch_" + k] = patch_weight_dict[k] * patch_loss_dict[k].cpu().detach().numpy()
      summary_loss_dict["patch_" + k] = patch_loss_dict[k].cpu().detach().numpy()

    if args.eval_res_cov:
      #point to mesh
      # print('patch logits shape: ', patch_predictions['closed_patch_logits'].shape)
      patch_close_logits = patch_predictions['closed_patch_logits'][0].detach().cpu().numpy()
      patch_close_logits = patch_close_logits[np.where(patch_labels == 0)]
      patch_uclosed = patch_close_logits[:,0] < patch_close_logits[:,1]
      
      if not args.eval_param:
        distances = compute_overall_singlecd(effective_patch_points, patch_uclosed, input_pointcloud)
      else:
        distances = compute_overall_singlecd_param(patch_predictions, input_pointcloud)
      
      summary_loss_dict["overall_single_cd"] = distances.mean()
      summary_loss_dict['p_cov_1'] = (distances < 0.01).sum() / input_pointcloud.shape[0]
      summary_loss_dict['p_cov_2'] = (distances < 0.02).sum() / input_pointcloud.shape[0]
      
      print('distance mean: {:06f} max: {:06f} min: {:06f} diff: {:06f} std:{:06f}'.format(distances.mean(), distances.max(), distances.min(), distances.max() - distances.min(), np.std(distances)))
      # print('distance shape: ', distances.shape)
      red_color = np.array([255, 0,0])
      white_color = np.array([255,255,255])
      th_dist = 0.01
      distances[distances<th_dist] = 0.0
      distances[distances>th_dist] = th_dist
      distances = distances.reshape([-1,1]) / th_dist
      pcolor = (1.0 - distances) * white_color + distances * red_color
      if not args.no_output:
        plywrite.save_vert_color_ply(input_pointcloud[:,:3], pcolor, os.path.join(obj_dir, "{}_pc_vis.ply".format(sample_id)))
        plywrite.save_vert_color_ply(closest_points, pcolor, os.path.join(obj_dir, "{}_closest_vis.ply".format(sample_id)))
        mesh.export(os.path.join(obj_dir, "{}_allpatch.obj".format(sample_id)))

      
    #accuracy of corner and curve predictions
    # summary_loss_dict['corner_valid_accuracy'] = corner_loss_dict['corner_prediction_accuracy'].cpu().detach().numpy()

    # summary_loss_dict['curve_valid_accuracy'] = curve_loss_dict['valid_class_accuracy'].cpu().detach().numpy()
    # summary_loss_dict['curve_type_accuracy'] = curve_loss_dict['type_class_accuracy'].cpu().detach().numpy()
    # summary_loss_dict['patch_valid_accuracy'] = patch_loss_dict['valid_class_accuracy'].cpu().detach().numpy()
    # summary_loss_dict['patch_type_accuracy'] = patch_loss_dict['type_class_accuracy'].cpu().detach().numpy()
    
    if not args.no_topo:
      if(args.curve_corner_geom_loss_coef > 0 or args.curve_corner_topo_loss_coef > 0):
        # curve_corner_matching_loss_geom, curve_corner_matching_loss_topo, all_zero_corners = \
        #     Curve_Corner_Matching_v2(corner_predictions, curve_predictions, target_corner_points_list, target_curves_list, corner_matching_indices['indices'], curve_matching_indices['indices'])
        # if args.ori_topo:
        #   curve_corner_matching_loss_geom, curve_corner_matching_loss_topo, all_zero_corners = \
        #     Curve_Corner_Matching_v2(corner_predictions, curve_predictions, target_corner_points_list, target_curves_list, corner_matching_indices['indices'], curve_matching_indices['indices'])
        # else:
        curve_corner_matching_loss_geom, curve_corner_matching_loss_topo, all_zero_corners = \
          Curve_Corner_Matching_tripath(corner_predictions, curve_predictions, target_corner_points_list, target_curves_list, corner_matching_indices['indices'], curve_matching_indices['indices'], flag_round = True)
        if(sample_count == 1): print("with curve corner correspondence loss")
        # try:
        #   # losses += args.curve_corner_geom_loss_coef*curve_corner_matching_loss_geom + args.curve_corner_topo_loss_coef*curve_corner_matching_loss_topo
        # except:
        #   print(args.curve_corner_geom_loss_coef*curve_corner_matching_loss_geom)
        #   print(args.curve_corner_topo_loss_coef*curve_corner_matching_loss_topo)
        #   print(losses)
        #   print(losses.shape)
        #   print((args.curve_corner_geom_loss_coef*curve_corner_matching_loss_geom).shape)
        #   raise Exception("Error")
        
        summary_loss_dict['corner_curve_topo'] = curve_corner_matching_loss_topo.cpu().detach().numpy()
        # summary_loss_dict['corner_curve_geom'] = args.curve_corner_geom_loss_coef*curve_corner_matching_loss_geom.cpu().detach().numpy()
        if('aux_outputs' in corner_predictions):
          assert('aux_outputs' in curve_predictions)
          for aux_id in range(len(corner_matching_indices['aux_outputs'])):
            curve_corner_matching_loss_geom_aux, curve_corner_matching_loss_topo_aux = \
              Curve_Corner_Matching_v2(corner_predictions['aux_outputs'][aux_id], curve_predictions['aux_outputs'][aux_id], target_corner_points_list, target_curves_list, corner_matching_indices['aux_outputs'][aux_id]['indices'], curve_matching_indices['aux_outputs'][aux_id]['indices'])
            # losses += args.curve_corner_geom_loss_coef*curve_corner_matching_loss_geom_aux + args.curve_corner_topo_loss_coef*curve_corner_matching_loss_topo_aux
            #summary_loss_dict['corner_curve_topo_aux_'+str(aux_id)] = args.curve_corner_topo_loss_coef*curve_corner_matching_loss_topo_aux.cpu().detach().numpy()
      elif(sample_count == 1):
        print("without curve corner correspondence loss")

      if(args.patch_curve_topo_loss_coef > 0):
        # patch_curve_matching_loss_topo = \
        #     Patch_Curve_Matching(curve_predictions, patch_predictions, target_curves_list, target_patches_list, curve_matching_indices['indices'], patch_matching_indices['indices'])
        # if args.ori_topo:
        #   patch_curve_matching_loss_topo = \
        #     Patch_Curve_Matching(curve_predictions, patch_predictions, target_curves_list, target_patches_list, curve_matching_indices['indices'], patch_matching_indices['indices'])
        # else:
        patch_curve_matching_loss_topo, p2p_loss_topo = \
          Patch_Curve_Matching_tripath(curve_predictions, patch_predictions, target_curves_list, target_patches_list, curve_matching_indices['indices'], patch_matching_indices['indices'], flag_round = True)  
        # losses += args.patch_curve_topo_loss_coef*patch_curve_matching_loss_topo
        summary_loss_dict['patch_curve_topo'] = patch_curve_matching_loss_topo.cpu().detach().numpy()
        summary_loss_dict['patch_patch_topo'] = p2p_loss_topo
        
        if('aux_outputs' in corner_predictions):
          assert('aux_outputs' in curve_predictions)
          for aux_id in range(len(corner_matching_indices['aux_outputs'])):
            patch_curve_matching_loss_topo_aux = \
              Patch_Curve_Matching(curve_predictions['aux_outputs'][aux_id], patch_predictions['aux_outputs'][aux_id], target_curves_list, target_patches_list, curve_matching_indices['aux_outputs'][aux_id]['indices'], patch_matching_indices['aux_outputs'][aux_id]['indices'])
            # losses += args.patch_curve_topo_loss_coef*patch_curve_matching_loss_topo_aux
      elif(sample_count == 1):
        print("without patch curve correspondence loss")
      
      if (args.patch_corner_topo_loss_coef > 0 and not args.ori_topo):
        patch_corner_matching_loss_topo, curve_point_loss, curve_patch_loss, patch_close_loss = \
          Patch_Corner_Matching_tripath(corner_predictions, curve_predictions, patch_predictions, target_corner_points_list, target_curves_list, target_patches_list, corner_matching_indices['indices'],curve_matching_indices['indices'], patch_matching_indices['indices'],flag_round = True)
      
        summary_loss_dict['patch_corner_topo'] = patch_corner_matching_loss_topo.cpu().detach().numpy()
        # summary_loss_dict['curve_point_loss'] = args.topo_loss_coef * curve_point_loss.cpu().detach().numpy()
        # summary_loss_dict['curve_patch_loss'] = args.topo_loss_coef * curve_patch_loss.cpu().detach().numpy()
        # summary_loss_dict['patch_close_loss'] = args.topo_loss_coef * patch_close_loss.cpu().detach().numpy()
        # summary_loss_dict['total_loss'] = losses.cpu().detach().numpy()
    filename = os.path.join(obj_dir, "{}_prediction.pkl".format(sample_id))
    output_prediction(filename, corner_predictions, curve_predictions, patch_predictions, \
      target_corner_points_list, target_curves_list, target_patches_list, corner_matching_indices, curve_matching_indices, patch_matching_indices, sample_id=sample_id)

    for k in summary_loss_dict:
      if k in dict_sum:
        dict_sum[k] += summary_loss_dict[k]
      else:
        dict_sum[k] = summary_loss_dict[k] + 0.0

    now = datetime.now()
    print("{} sample:{}".format(now, sample_id))
    # print(summary_loss_dict)
    
    sample_name_list.append(sample_id)
    test_statistics.append(list(summary_loss_dict.values()))
    
    #train_summary = tf_summary_from_dict(summary_loss_dict, True)
    #summary_writer.add_summary(tf.compat.v1.Summary(value=train_summary), train_iter)
    
    # reduce losses over all GPUs for logging purposes
    # loss_dict_reduced = reduce_dict(corner_loss_dict)
    # #loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
    # loss_dict_reduced_scaled = {k: v * corner_weight_dict[k] for k, v in loss_dict_reduced.items() if k in corner_weight_dict}
    # losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
    # loss_value = losses_reduced_scaled.item()
  
    # if not math.isfinite(loss_value):
    #     print("Corner Loss is {}, stopping training".format(loss_value))
    #     print(loss_dict_reduced)
    #     print(batch_sample_id)
    #     sys.exit(1)
    
    # loss_dict_reduced = reduce_dict(curve_loss_dict)
    # #loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
    # loss_dict_reduced_scaled = {k: v * curve_weight_dict[k] for k, v in loss_dict_reduced.items() if k in curve_weight_dict}
    # losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
    # loss_value = losses_reduced_scaled.item()
  
    # if not math.isfinite(loss_value):
    #     print("Loss is {}, stopping training".format(loss_value))
    #     print(loss_dict_reduced)
    #     print(batch_sample_id)
    #     sys.exit(1)
    
  # print(summary_loss_dict.keys())
  for k in dict_sum:
    dict_sum[k] = dict_sum[k] / sample_count

  sample_name_list.append("mean")
  test_statistics.append(list(dict_sum.values()))

  # test_statistics.append(list(summary_loss_dict.values()))


  np.savetxt(os.path.join(obj_dir, "test_statistics.txt"), np.array(test_statistics))
  with open(os.path.join(obj_dir, "test_sample_name.txt"), "w") as wf:
    for sample_name in sample_name_list:
      wf.write("{}\n".format(sample_name))
  print(sample_count, "samples in test set")
  
  import pandas as pd
  ## convert your array into a dataframe
  df = pd.DataFrame(np.array(test_statistics))
  df.columns = list(summary_loss_dict.keys())
  df.index = sample_name_list
  #title_row = pd.Series()
  #row_df = pd.DataFrame([title_row])
  #df = pd.concat([row_df, df], ignore_index=True)
  
  ## save to xlsx file
  filepath = os.path.join(obj_dir, 'test_statistics.xlsx')
  df.to_excel(filepath, index=True)

def model_evaluation_from_file(model_shape, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion, train_data, device, train_iter, flag_output = True):
  import trimesh
  from numpy import linalg as LA
  from scipy.spatial import cKDTree
  def numpy_chamfer_dist(src, tgt):
    kdtree1 = cKDTree(src)
    dist_tgt, idx_tgt = kdtree1.query(tgt)
    kdtree2 = cKDTree(tgt)
    dist_src, idx_src = kdtree2.query(src)
    return (dist_src.mean() + dist_tgt.mean()) / 2.0
  
  disable_aux_loss_output = True
  model_shape.eval()
  corner_loss_criterion.eval()
  curve_loss_criterion.eval()
  patch_loss_criterion.eval()
  assert(args.batch_size == 1)
  #test_data = train_data_loader(1, voxel_dim=voxel_dim, device=device, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, data_folder="test_data")
  data_loader_iterator = iter(train_data)
  
  obj_dir = os.path.join("experiments", args.experiment_name, "test_obj")
  if(not os.path.exists(obj_dir)): os.mkdir(obj_dir)
  
  test_statistics = []
  sample_name_list = []
  
  # sample_count = 0
  def export_curves(curve_points, points_number_per_curve, output_obj_file):
        curve_points = np.reshape(curve_points, [-1,3])
        with open(output_obj_file, "w") as wf:
          for point in curve_points:
            wf.write("v {} {} {}\n".format(point[0], point[1], point[2]))
          for i in range(curve_points.shape[0]):
            if(i % points_number_per_curve == (points_number_per_curve - 1)):
              continue
            wf.write("l {} {}\n".format(i+1, i+2))

  dict_sum = {}

  dict_sum["corner_all_cd"] = 0.0
  dict_sum["curve_all_cd"] = 0.0
  dict_sum["patch_all_cd"] = 0.0

  #valid data from test dictionary
  path = obj_dir
  allfs = os.listdir(obj_dir)
  all_valid_id = []
  for f in allfs:
    if f.endswith("_opt_post_mix_final_curves.ply"):
      all_valid_id.append(int(f.split("_")[0]))
  
  print('valid size: ', len(all_valid_id))
  all_valid_id_set = set(all_valid_id)
  
  nopatchcount = 0
  while(True):
    try:
      data_item = next(data_loader_iterator)
    except StopIteration:
      #data_loader_iterator = iter(train_data)
      #data_item = next(data_loader_iterator)
      break
    locations = data_item[0].to(device)
    features = data_item[1].to(device)
    corner_points = data_item[2].to(device)
    corner_batch_idx = data_item[3].to(device)
    batch_sample_id = data_item[5]
    target_curves_list = data_item[6]
    target_patches_list = data_item[7]

    sample_id = int(batch_sample_id[0].replace(".pkl" ,""))
    if sample_id not in all_valid_id_set:
      continue
    # sample_count+=1
    print ('sample_id: ', sample_id)
    #convert target_curves_list and target_patches_list to cuda tensors
    for i in range(len(target_curves_list)):
      for k in target_curves_list[i]:
        target_curves_list[i][k] = target_curves_list[i][k].to(device)
    
    for i in range(len(target_patches_list)):
      for k in target_patches_list[i]:
        if k == 'patch_points' or k == 'patch_normals':
          for j in range(len(target_patches_list[i][k])): #list
            target_patches_list[i][k][j] = target_patches_list[i][k][j].to(device)
        else:
          target_patches_list[i][k] = target_patches_list[i][k].to(device)
    
    #supervision
    batch_corner_numbers = []
    for i in range(args.batch_size):
      batch_corner_numbers.append((corner_batch_idx==i).sum())
    target_corner_points_list = torch.split(corner_points, batch_corner_numbers)
    
    #batch size should be one
    cur_id = sample_id
    pred_curve_file = os.path.join(path, "{}_opt_post_mix_final_curves.ply".format(cur_id))
    pred_curve_close_file = os.path.join(path, "{}_opt_post_mix_final_curves_close.txt".format(cur_id))
    pred_corner_file = os.path.join(path, "{}_opt_post_mix_final_corner.ply".format(cur_id))
    pred_patch_file = os.path.join(path, "{}_opt_post_mix_step5.obj".format(cur_id))
    pred_patch_mask_file = os.path.join(path, "{}_opt_post_mix_facemask.txt".format(cur_id))
    pred_corres = os.path.join(path, "{}_opt_post_mix_final_topo.txt".format(cur_id))
    if args.fittingonce:
      pred_curve_file = os.path.join(path, "{}_opt_post_mix_curves_init.ply".format(cur_id))
      pred_corner_file = os.path.join(path, "{}_opt_post_mix_corners_init.ply".format(cur_id))
    
    if not os.path.exists(pred_patch_mask_file):
      continue
    if not os.path.exists(pred_corres):
      continue

    pred_curve = load_ply_pts(pred_curve_file)
    pred_corner = load_ply_pts(pred_corner_file)
    
    #numpy corner curve patch
    # curve = target_curves_list[0]['curve_points'][::3].cpu().numpy()
    curve = []
    for i in range(target_curves_list[0]['curve_points'].shape[0]):
      curve.append(target_curves_list[0]['curve_points'][i].cpu().numpy())
    curve = np.concatenate(curve)
    corner = target_corner_points_list[0].cpu().numpy()
    allpatch = []
    for i in range(len(target_patches_list)):
      for j in range(len(target_patches_list[i]['patch_points'])): #list
        allpatch.append(target_patches_list[i]['patch_points'][j].cpu().numpy())

    allpatch = np.concatenate(allpatch)
    # print("all patch shape ", allpatch.shape)        
    dist_curve = 0.0
    if pred_curve.shape[0] > 0 and curve.shape[0] > 0:
      dist_curve = numpy_chamfer_dist(pred_curve, curve)
    dist_corner = 0.0
    if pred_corner.shape[0] > 0 and corner.shape[0] > 0:
      dist_corner = numpy_chamfer_dist(pred_corner, corner)
    
    dist_patch = 0.0
    n_sample = 10000
    # print('mesh file: ', pred_patch_file)
    mesh = trimesh.load(pred_patch_file)
    #if there are no faces, set distance to 0
    meshtype = str(type(mesh))
    pc, idx = None, None
    if meshtype.split(' ')[-1].split('.')[-1][:-2] != 'PointCloud':
      pc, idx = mesh.sample(n_sample, return_index=True)
      dist_patch = numpy_chamfer_dist(pc, allpatch)
      # np.savetxt(os.path.join(path, "{}_opt_post_mix_step5.xyz".format(cur_id)), pc)
      # np.savetxt(os.path.join(path, "{}_opt_post_mix_gt.xyz".format(cur_id)), allpatch[:, :3])
      # np.savetxt('gt_patch.xyz', allpatch[:, :3])
    else:
      nopatchcount += 1
    
    summary_loss_dict = {}
    summary_loss_dict["corner_all_cd"] = dist_corner
    summary_loss_dict["curve_all_cd"] = dist_curve
    summary_loss_dict["patch_all_cd"] = dist_patch
    
    pred_face_mask = np.loadtxt(pred_patch_mask_file, dtype = int)
    pred_curve_close = np.loadtxt(pred_curve_close_file, dtype = int)
    #get pred data
    pred_corner_data = {}
    pred_corner_data['pred_corner_position'] = torch.from_numpy(pred_corner).type(torch.FloatTensor).unsqueeze(0).to(device)
    pred_corner_data['pred_logits'] = torch.zeros([1,pred_corner.shape[0], 2], device = device)
    pred_corner_data['pred_logits'][0][:,0] = 1
    
    pred_curve_data = {}
    pred_curve_data['pred_curve_points'] = torch.from_numpy(pred_curve).type(torch.FloatTensor).view(1,-1,34,3).to(device)
    # print('pred curve points, ', pred_curve_data['pred_curve_points'].shape)
    # print('gt curve points, ', target_curves_list[0]['curve_points'].shape)
    pred_curve_data['pred_curve_logits'] = torch.zeros([1,pred_curve.shape[0] // 34, 2], device = device)
    pred_curve_data['pred_curve_logits'][0][:,0] = 1
    pred_curve_data['pred_curve_type'] = torch.zeros([1,pred_curve.shape[0], 4], device = device) #not set
    
    pred_curve_data['closed_curve_logits'] = torch.zeros([1,pred_curve.shape[0], 2], device = device) #not need to set
    
    #length weighting set to one
    target_curves_list[0]['curve_length_weighting'][:] = 1.0

    pred_patch_data = {}
    #patch points

    #corner matching
    corner_loss_dict, corner_matching_indices = corner_loss_criterion(pred_corner_data, target_corner_points_list)

    curve_loss_dict, curve_matching_indices = curve_loss_criterion(pred_curve_data, target_curves_list)

    
    #get topo information
    for k in corner_loss_dict.keys():
      summary_loss_dict["corner_" + k] = corner_loss_dict[k].item()
    
    for k in curve_loss_dict.keys():
      summary_loss_dict["curve_" + k] = curve_loss_dict[k].item()
    
    n_pred_patch = 0
    patch_samples = []
    if pred_face_mask.shape[0]:
      n_pred_patch = pred_face_mask.max() + 1
      mesh_verts = mesh.vertices
      mesh_faces = mesh.faces
      patch_faces = [[] for i in range(n_pred_patch)]
      assert(mesh_faces.shape[0] == pred_face_mask.shape[0])
      for i in range(pred_face_mask.shape[0]):
        patch_faces[pred_face_mask[i]].append(mesh_faces[i])
      
      n_single_patch_sample = 100
      flag_emtpy_patch = False
      for i in range(n_pred_patch):
        # if len(patch_faces[i]) == 0:
        #   print("empty patch: ", i)
        #   flag_emtpy_patch = True
        #   break
        # commented on 0903
        assert(len(patch_faces[i]) > 0)
        tmpmesh = trimesh.Trimesh(vertices=mesh_verts, faces = patch_faces[i])
        tmppc, tmpidx = tmpmesh.sample(n_single_patch_sample, return_index=True)
        patch_samples.append(tmppc)
      # np.savetxt('patch_{}.xyz'.format(i), tmppc)
      if flag_emtpy_patch:
        continue
      pred_patch_data['pred_patch_points'] = torch.tensor(patch_samples,dtype=torch.float32, device = device).view(1,-1, 100, 3)
      pred_patch_data['pred_patch_logits'] = torch.zeros([1,n_pred_patch,2],device = device)
      pred_patch_data['pred_patch_logits'][0][:,0] = 1
      pred_patch_data['pred_patch_type'] = torch.zeros([1,n_pred_patch,6], device = device)
      patch_loss_dict, patch_matching_indices = patch_loss_criterion(pred_patch_data, target_patches_list)
      for k in patch_loss_dict.keys():
        summary_loss_dict["patch_" + k] = patch_loss_dict[k].item()
    else:
      summary_loss_dict["patch_cd"] = 0.0
      summary_loss_dict["patch_precision"] = 0.0
      summary_loss_dict["patch_recall"] = 0.0
      summary_loss_dict["patch_type_class_accuracy"] = 0.0

    
    #topo error
    f = open(pred_corres, 'r')
    lines = f.readlines()
    line_iter = 0
    n_topo_patch = int(lines[0].split()[0])
    n_topo_curve = int(lines[0].split()[1])
    line_iter += 1
    pred_patch2curve = [[] for i in range(n_topo_patch)]
    for i in range(n_topo_patch):
      line = lines[line_iter].split()
      assert(len(line) == n_topo_curve)
      for j in range(n_topo_curve):
        pred_patch2curve[i].append(float(line[j]))
      line_iter += 1

    n_topo_corner = int(lines[line_iter].split()[1])
    print("patch {} curve {} corner {}".format(n_topo_patch, n_topo_curve, n_topo_corner))
    pred_curve2corner = [[] for i in range(n_topo_curve)]
    line_iter += 1
    for i in range(n_topo_curve):
      line = lines[line_iter].split()
      assert(len(line) == n_topo_corner)
      for j in range(n_topo_corner):
        pred_curve2corner[i].append(float(line[j]))
      line_iter += 1
    f.close()
    
    
    if patch_matching_indices['indices'][0][0].shape[0] and curve_matching_indices['indices'][0][0].shape[0]:
      #patch curve correspondence
      pred_patch2curve = torch.tensor(pred_patch2curve, device = device, dtype = torch.float).view(len(pred_patch2curve), -1)

      pred_patch2curve = pred_patch2curve[patch_matching_indices['indices'][0][0],][:,curve_matching_indices['indices'][0][0]]
      gt_patch2curve = target_patches_list[0]['patch_curve_correspondence'][patch_matching_indices['indices'][0][1],][:,curve_matching_indices['indices'][0][1]]
      assert(pred_patch2curve.shape == gt_patch2curve.shape)
      # summary_loss_dict['topo_patch_curve'] = F.binary_cross_entropy(pred_patch2curve.view(-1), gt_patch2curve.view(-1)).item()
      summary_loss_dict['topo_patch_curve'] = (pred_patch2curve - gt_patch2curve).abs().mean().item()
      # summary_loss_dict['topo_patch_curve'] /= (len(pred_patch2curve) * (len(pred_patch2curve[0])))
    else:
      summary_loss_dict['topo_patch_curve'] = 0.0
    
    if curve_matching_indices['indices'][0][0].shape[0] and corner_matching_indices['indices'][0][0].shape[0]:
      pred_curve2corner = torch.tensor(pred_curve2corner, device=device, dtype = torch.float).view(len(pred_curve2corner), -1)
      pred_curve2corner = pred_curve2corner[curve_matching_indices['indices'][0][0],][:,corner_matching_indices['indices'][0][0]]
      
      #not filtered with open curve idx
      curve2corner_gt = target_curves_list[0]['endpoints'][curve_matching_indices['indices'][0][1]]
      # curve2corner_gt = target_curves_list[0]['endpoints']

      gt_curve2corner = torch.zeros([curve_matching_indices['indices'][0][1].shape[0], target_corner_points_list[0].shape[0]], device=device, dtype = torch.float)
      
      # print('gt curve2corner : ', gt_curve2corner.shape)
      # print('curve2corner: ', curve2corner_gt)
      gt_curve2corner[torch.arange(gt_curve2corner.shape[0]), curve2corner_gt[:,0]] = 1
      gt_curve2corner[torch.arange(gt_curve2corner.shape[0]), curve2corner_gt[:,1]] = 1

      for i in range(curve_matching_indices['indices'][0][1].shape[0]):
        ori_curveid = curve_matching_indices['indices'][0][1][i]
        if (target_curves_list[0]['is_closed'][ori_curveid] >= 0.5):
          gt_curve2corner[i] = 0.0

      gt_curve2corner = gt_curve2corner[:, corner_matching_indices['indices'][0][1]]

      assert(pred_curve2corner.shape == gt_curve2corner.shape)
      # summary_loss_dict['topo_curve_corner'] = F.binary_cross_entropy(pred_curve2corner.view(-1), gt_curve2corner.view(-1)).item()
      summary_loss_dict['topo_curve_corner'] = (pred_curve2corner - gt_curve2corner).abs().mean().item()
      
    else:
      summary_loss_dict['topo_curve_corner'] = 0.0
    
    for k in summary_loss_dict:
      if k in dict_sum:
        dict_sum[k] += summary_loss_dict[k]
      else:
        dict_sum[k] = summary_loss_dict[k]

    # print(list(summary_loss_dict.values()))
    assert(len(summary_loss_dict) == 16)
    test_statistics.append(list(summary_loss_dict.values()))
    # sample_id.append(cur_id)
    sample_name_list.append(cur_id)
  for k in dict_sum:
    dict_sum[k] = dict_sum[k] / (len(sample_name_list))

  print('dict sum: ', dict_sum)
  sample_name_list.append("mean")
  test_statistics.append(list(dict_sum.values()))

  # test_statistics.append(list(summary_loss_dict.values()))

  # obj
  np.savetxt(os.path.join(obj_dir, "final_evaluation.txt"), np.array(test_statistics))
  with open(os.path.join(obj_dir, "test_sample_name.txt"), "w") as wf:
    for sample_name in sample_name_list:
      wf.write("{}\n".format(sample_name))
  print(len(sample_name_list) - 1, "samples in test set")
  
  import pandas as pd
  ## convert your array into a dataframe
  df = pd.DataFrame(np.array(test_statistics))
  # df.columns = list(summary_loss_dict.keys())
  df.columns = list(dict_sum.keys())
  df.index = sample_name_list
  #title_row = pd.Series()
  #row_df = pd.DataFrame([title_row])
  #df = pd.concat([row_df, df], ignore_index=True)
  
  ## save to xlsx file
  filepath = os.path.join(obj_dir, 'final_evaluation.xlsx')
  df.to_excel(filepath, index=True)

def load_complex_file(fn, device):
    n_curve_sample = 34
    n_patch_sample = 100
    pred_data = {}
    f = open(fn, 'r')
    lines = f.readlines()    
    f.close()
    lineid = 0
    line = lines[0].split()
    first_elem = int(line[0])
    if first_elem != -1:
      n_corner = int(line[0])
      n_curve = int(line[1])
      n_patch = int(line[2])
      lineid += 1
    else:
      lineid += 1
      n_patch_sample = 400
      line = lines[lineid].split()
      n_corner = int(line[0])
      n_curve = int(line[1])
      n_patch = int(line[2])
      lineid += 1
    
    #corner
    pred_data['pred_corner_position'] = torch.zeros([1, n_corner, 3], device = device)
    pred_data['pred_logits'] = torch.zeros([1, n_corner, 2], device = device)
    pred_data['pred_logits'][0][:,0] = 1.0

    for i in range(n_corner):
      line = lines[lineid].split()
      for j in range(3):
        pred_data['pred_corner_position'][0][i][j] = float(line[j])
      lineid += 1

    #curve
    pred_data['pred_curve_logits'] = torch.zeros([1, n_curve, 2], device = device)
    pred_data['pred_curve_logits'][0][:,0] = 1
    pred_data['pred_curve_type'] = torch.zeros([1, n_curve, 4], device = device) 
    
    pred_data['closed_curve_logits'] = torch.zeros([1, n_curve, 2], device = device)
    pred_data['pred_curve_points'] = torch.zeros([1,n_curve, n_curve_sample, 3], device =device)
    for i in range(n_curve):
      line = lines[lineid].split()
      for j in range(n_curve_sample):
        for k in range(3):
          pred_data['pred_curve_points'][0][i][j][k] = float(line[2 + 3 * j + k])
      pred_data['pred_curve_type'][0][i][curve_type_to_id(line[0])] = 1.0
      # if float(line[1]) > 0.5:
      #   pred_data['closed_curve_logits'][0][i][1] = 1.0
      # else:
      #   pred_data['closed_curve_logits'][0][i][0] = 1.0
      pred_data['closed_curve_logits'][0][i][1] = float(line[1])
      pred_data['closed_curve_logits'][0][i][0] = 1 - float(line[1])

      lineid += 1
    
    #patch
    # print('corner: {} curve: {} patch: {}'.format(n_corner, n_curve, n_patch))
    pred_data['pred_patch_points'] = torch.zeros([1, n_patch, n_patch_sample, 3], device = device)
    pred_data['pred_patch_logits'] = torch.zeros([1, n_patch, 2], device=device)
    pred_data['pred_patch_logits'][0][:,0] = 1
    pred_data['pred_patch_type'] = torch.zeros([1,n_patch, 6], device = device)
    pred_data['closed_patch_logits'] = torch.zeros([1, n_patch, 2], device = device) 
    

    for i in range(n_patch):
      line = lines[lineid].split()
      for j in range(n_patch_sample):
        for k in range(3):
          pred_data['pred_patch_points'][0][i][j][k] = float(line[1 + 3 * j + k])
      pred_data['pred_patch_type'][0][i][patch_type_to_id(line[0])] = 1.0
      lineid += 1
    
    sim_curvecorner = [[] for i in range(n_curve)]
    for i in range(n_curve):
      line = lines[lineid].split()
      assert(len(line) == n_corner)
      for j in range(n_corner):
        sim_curvecorner[i].append(float(line[j]))
      lineid += 1

    sim_patchcurve = [[] for i in range(n_patch)]
    for i in range(n_patch):
      line = lines[lineid].split()
      assert(len(line) == n_curve)
      for j in range(n_curve):
        sim_patchcurve[i].append(float(line[j]))
      lineid += 1
    pred_data['patch2curve'] = sim_patchcurve
    pred_data['curve2corner'] = sim_curvecorner
    
    #patch_uclosed
    if lineid < len(lines):
      for i in range(n_patch):
        line = lines[lineid].split()
        pred_data['closed_patch_logits'][0][i][0] = 1.0 - float(line[0])
        pred_data['closed_patch_logits'][0][i][1] = float(line[0])
        lineid += 1
    else:
      print("no patch close info")

    sim_patchcorner = [[] for i in range(n_patch)]
    for i in range(n_patch):
      line = lines[lineid].split()
      assert(len(line) == n_corner)
      for j in range(n_corner):
        sim_patchcorner[i].append(float(line[j]))
      lineid += 1
    pred_data['patch2corner'] = sim_patchcorner


    return pred_data

def model_evaluation_topo(model_shape, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion, train_data, device, train_iter, flag_output = True):
  import trimesh
  from numpy import linalg as LA
  from scipy.spatial import cKDTree
  
  #not used tmporarily
  
  # suffix = "_opt_post_mix.complex"
  suffix = "_opt_mix.complex"
  # suffix = "_prediction.complex"
  # suffix = "_gt.complex"

  disable_aux_loss_output = True
  model_shape.eval()
  corner_loss_criterion.eval()
  curve_loss_criterion.eval()
  patch_loss_criterion.eval()
  assert(args.batch_size == 1)
  #test_data = train_data_loader(1, voxel_dim=voxel_dim, device=device, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, data_folder="test_data")
  data_loader_iterator = iter(train_data)
  
  # obj_dir = os.path.join("experiments", args.experiment_name, "test_obj")
  # obj_dir = os.path.join("experiments", args.experiment_name, "256_nms_pred")
  obj_dir = os.path.join("experiments", args.experiment_name, "exp_81_val03_pred")
  # obj_dir = os.path.join("experiments", args.experiment_name, "256_nms_mix_patchcorner")
  # obj_dir = os.path.join("experiments", args.experiment_name, "nms_mix_81")
  # obj_dir = os.path.join("experiments", args.experiment_name, "nms_pred_81")
  # obj_dir = os.path.join("experiments", args.experiment_name, "twomodel")



  if args.vis_test:
    obj_dir = os.path.join("experiments", args.experiment_name, "vis_test")
  if args.vis_train:
    obj_dir = os.path.join("experiments", args.experiment_name, "vis_train")
  if(not os.path.exists(obj_dir)): os.mkdir(obj_dir)
  
  test_statistics = []
  sample_name_list = []
  
  # sample_count = 0
  def export_curves(curve_points, points_number_per_curve, output_obj_file):
        curve_points = np.reshape(curve_points, [-1,3])
        with open(output_obj_file, "w") as wf:
          for point in curve_points:
            wf.write("v {} {} {}\n".format(point[0], point[1], point[2]))
          for i in range(curve_points.shape[0]):
            if(i % points_number_per_curve == (points_number_per_curve - 1)):
              continue
            wf.write("l {} {}\n".format(i+1, i+2))

  dict_sum = {}

  #valid data from test dictionary
  path = obj_dir
  allfs = os.listdir(obj_dir)
  all_valid_id = []
  for f in allfs:
    if f.endswith(suffix):
      all_valid_id.append(f.replace(suffix, ""))
  
  all_valid_id_set = set(all_valid_id)
  
  nopatchcount = 0
  while(True):
    try:
      data_item = next(data_loader_iterator)
    except StopIteration:
      #data_loader_iterator = iter(train_data)
      #data_item = next(data_loader_iterator)
      break
    locations = data_item[0].to(device)
    features = data_item[1].to(device)
    corner_points = data_item[2].to(device)
    corner_batch_idx = data_item[3].to(device)
    batch_sample_id = data_item[5]
    target_curves_list = data_item[6]
    target_patches_list = data_item[7]
    input_pointcloud = data_item[4][0]

    sample_id = batch_sample_id[0].replace("_fix.pkl" ,"")
    print ('sample_id: ', sample_id)
    if sample_id not in all_valid_id_set:
      continue
    # sample_count+=1
    #convert target_curves_list and target_patches_list to cuda tensors
    for i in range(len(target_curves_list)):
      for k in target_curves_list[i]:
        target_curves_list[i][k] = target_curves_list[i][k].to(device)
    
    for i in range(len(target_patches_list)):
      for k in target_patches_list[i]:
        if k == 'patch_points' or k == 'patch_normals' or k == 'patch_pcs':
          for j in range(len(target_patches_list[i][k])): #list
            target_patches_list[i][k][j] = target_patches_list[i][k][j].to(device)
        else:
          # print('k: ', k)
          target_patches_list[i][k] = target_patches_list[i][k].to(device)
    
    #supervision
    batch_corner_numbers = []
    for i in range(args.batch_size):
      batch_corner_numbers.append((corner_batch_idx==i).sum())
    target_corner_points_list = torch.split(corner_points, batch_corner_numbers)
    
    #batch size should be one
    cur_id = sample_id
    complex_file = os.path.join(path, "{}{}".format(cur_id, suffix))
    pred_data = load_complex_file(complex_file)

    summary_loss_dict = {}
    #corner matching
    corner_loss_dict, corner_matching_indices = corner_loss_criterion(pred_data, target_corner_points_list)
    curve_loss_dict, curve_matching_indices = curve_loss_criterion(pred_data, target_curves_list)

    
    #get topo information
    for k in corner_loss_dict.keys():
      summary_loss_dict["corner_" + k] = corner_loss_dict[k].item()
    
    for k in curve_loss_dict.keys():
      summary_loss_dict["curve_" + k] = curve_loss_dict[k].item()
    
    target_curves_list[0]['curve_length_weighting'][:] = 1.0

    patch_loss_dict, patch_matching_indices = patch_loss_criterion(pred_data, target_patches_list)
    for k in patch_loss_dict.keys():
      summary_loss_dict["patch_" + k] = patch_loss_dict[k].item()

    if args.eval_res_cov:
      patch_close_logits = pred_data['closed_patch_logits'][0].detach().cpu().numpy()
      patch_uclosed = patch_close_logits[:,0] < patch_close_logits[:,1]
      # distances = compute_overall_singlecd(pred_data['pred_patch_points'][0].detach().cpu().numpy(), patch_uclosed, input_pointcloud)
      if not args.eval_param:
        distances = compute_overall_singlecd(pred_data['pred_patch_points'][0].detach().cpu().numpy(), patch_uclosed, input_pointcloud)
      else:
        distances = compute_overall_singlecd_param(pred_data, input_pointcloud)
      
      summary_loss_dict["overall_single_cd"] = distances.mean()
      summary_loss_dict['p_cov_1'] = (distances < 0.01).sum() / input_pointcloud.shape[0]
      summary_loss_dict['p_cov_2'] = (distances < 0.02).sum() / input_pointcloud.shape[0]
      
    
    #topo error
    pred_patch2curve = pred_data['patch2curve'] #lists of lists
    pred_curve2corner = pred_data['curve2corner'] #lists of lists
    
    if patch_matching_indices['indices'][0][0].shape[0] and curve_matching_indices['indices'][0][0].shape[0]:
      #patch curve correspondence
      pred_patch2curve = torch.tensor(pred_patch2curve, device = device, dtype = torch.float).view(len(pred_patch2curve), -1)

      pred_patch2curve = pred_patch2curve[patch_matching_indices['indices'][0][0],][:,curve_matching_indices['indices'][0][0]]
      
      gt_patch2curve = target_patches_list[0]['patch_curve_correspondence'][patch_matching_indices['indices'][0][1],][:,curve_matching_indices['indices'][0][1]]
      assert(pred_patch2curve.shape == gt_patch2curve.shape)
      # summary_loss_dict['topo_patch_curve'] = F.binary_cross_entropy(pred_patch2curve.view(-1), gt_patch2curve.view(-1)).item()
      summary_loss_dict['topo_patch_curve'] = (pred_patch2curve - gt_patch2curve).abs().mean().item()
      # summary_loss_dict['topo_patch_curve'] /= (len(pred_patch2curve) * (len(pred_patch2curve[0])))

      #here to compute patch-patch connection
      pred_patch2patch = torch.mm(pred_patch2curve, torch.transpose(pred_patch2curve, 0, 1))
      gt_patch2patch = torch.mm(gt_patch2curve, torch.transpose(gt_patch2curve, 0, 1))
      # print("gt p2p: \n", gt_patch2patch)
      
      tmpid = torch.arange(patch_matching_indices['indices'][0][0].shape[0], device=pred_patch2patch.device)
      # assert(gt_patch2patch[tmpid, tmpid].min() > 0)
      pred_patch2patch[tmpid, tmpid] = 0
      gt_patch2patch[tmpid, tmpid] = 0
      pred_patch2patch[pred_patch2patch > 0.5] = 1
      gt_patch2patch[gt_patch2patch > 0.5] = 1
      
      # print("pred p2p: \n", pred_patch2patch)
      # print("gt p2p: \n", gt_patch2patch)
      # print("pred p2c: \n", pred_patch2curve)
      # print("gt p2c: \n", gt_patch2curve)
      num_match_patches = pred_patch2patch.shape[0]
      num_gt_patches = len(target_patches_list[0]['u_closed'])
      num_gt_curves = len(target_curves_list[0]['is_closed'])
      # print('match num: {} all num: {}'.format(num_match_patches, num_gt_patches))
      # print('curve gt num: ', num_gt_curves)
      summary_loss_dict['topo_patch_patch'] = ((pred_patch2patch - gt_patch2patch).abs().sum().item() + num_gt_patches * num_gt_patches - num_match_patches * num_match_patches) / (num_gt_patches * num_gt_patches)

    else:
      summary_loss_dict['topo_patch_curve'] = 0.0
      #to be changed, based on gt size
      num_gt_patches = len(target_patches_list[0]['u_closed'])
      num_gt_curves = len(target_curves_list[0]['is_closed'])
      if num_gt_patches == 0 or num_gt_curves == 0:
        summary_loss_dict['topo_patch_patch'] = 0.0
      else:
        summary_loss_dict['topo_patch_patch'] = 1.0
    
    if curve_matching_indices['indices'][0][0].shape[0] and corner_matching_indices['indices'][0][0].shape[0]:
      pred_curve2corner = torch.tensor(pred_curve2corner, device=device, dtype = torch.float).view(len(pred_curve2corner), -1)
      pred_curve2corner = pred_curve2corner[curve_matching_indices['indices'][0][0],][:,corner_matching_indices['indices'][0][0]]
      
      cur_curves_gt = target_curves_list[0]
      
      open_curve_idx = torch.where(cur_curves_gt['is_closed'][curve_matching_indices['indices'][0][1]] < 0.5)

      curve2corner_gt = target_curves_list[0]['endpoints'][curve_matching_indices['indices'][0][1]]
      # curve2corner_gt = target_curves_list[0]['endpoints']

      gt_curve2corner = torch.zeros([curve_matching_indices['indices'][0][1].shape[0], target_corner_points_list[0].shape[0]], device=device, dtype = torch.float)
      
      # print('gt curve2corner : ', gt_curve2corner.shape)
      # print('curve2corner: ', curve2corner_gt)
      gt_curve2corner[torch.arange(gt_curve2corner.shape[0]), curve2corner_gt[:,0]] = 1
      gt_curve2corner[torch.arange(gt_curve2corner.shape[0]), curve2corner_gt[:,1]] = 1

      for i in range(curve_matching_indices['indices'][0][1].shape[0]):
        ori_curveid = curve_matching_indices['indices'][0][1][i]
        if (target_curves_list[0]['is_closed'][ori_curveid] >= 0.5):
          gt_curve2corner[i] = 0.0

      gt_curve2corner = gt_curve2corner[:, corner_matching_indices['indices'][0][1]]
      gt_curve2corner = gt_curve2corner[open_curve_idx]
      pred_curve2corner = pred_curve2corner[open_curve_idx]
      assert(pred_curve2corner.shape == gt_curve2corner.shape)
      # summary_loss_dict['topo_curve_corner'] = F.binary_cross_entropy(pred_curve2corner.view(-1), gt_curve2corner.view(-1)).item()
      summary_loss_dict['topo_curve_corner'] = (pred_curve2corner - gt_curve2corner).abs().mean().item()
      
    else:
      summary_loss_dict['topo_curve_corner'] = 0.0
    
    for k in summary_loss_dict:
      if k in dict_sum:
        dict_sum[k] += summary_loss_dict[k]
      else:
        dict_sum[k] = summary_loss_dict[k]

    # print(list(summary_loss_dict.values()))
    # assert(len(summary_loss_dict) == 13)
    test_statistics.append(list(summary_loss_dict.values()))
    # sample_id.append(cur_id)
    sample_name_list.append(cur_id)
  for k in dict_sum:
    dict_sum[k] = dict_sum[k] / (len(sample_name_list))

  print('dict sum: ', dict_sum)
  sample_name_list.append("mean")
  test_statistics.append(list(dict_sum.values()))

  # obj
  # np.savetxt(os.path.join(obj_dir, "topo_evaluation.txt"), np.array(test_statistics)) #1118
  with open(os.path.join(obj_dir, "test_topo_sample_name.txt"), "w") as wf:
    for sample_name in sample_name_list:
      wf.write("{}\n".format(sample_name))
  print(len(sample_name_list) - 1, "samples in test set")
  
  import pandas as pd
  ## convert your array into a dataframe
  df = pd.DataFrame(np.array(test_statistics))
  # df.columns = list(summary_loss_dict.keys())
  df.columns = list(dict_sum.keys())
  df.index = sample_name_list
  #title_row = pd.Series()
  #row_df = pd.DataFrame([title_row])
  #df = pd.concat([row_df, df], ignore_index=True)
  
  ## save to xlsx file
  filepath = os.path.join(obj_dir, 'topo_evaluation.xlsx')
  df.to_excel(filepath, index=True)


def model_evaluation_yaml(model_shape, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion, train_data, device, train_iter, flag_output = True):
  import trimesh
  from numpy import linalg as LA
  from scipy.spatial import cKDTree
  import yaml
  import json
  with_curve_corner = True
  flag_with_param = args.eval_param
  # suffix = "_opt_mix_final.yml"
  # suffix = "_prediction_final.yml"
  # suffix = "_prediction_input.json"
  # suffix = "_opt_mix_final.json"
  suffix = args.suffix
  # suffix = "_prediction.complex"

  # suffix = "_prediction_final.json"



  # suffix = "_opt_post_mix.complex"
  # suffix = "_prediction_input.yml"
  # suffix = "_opt_post_mix_final.yml"
  # suffix = "_opt_mix_input.yml"
  
  # suffix = "_opt_post_mix_input.yml"
  # suffix = "_gt.complex"
  def load_yaml_json_file(fn):
    n_curve_sample = 34
    n_patch_sample = 100
    pred_data = {}
    if fn.endswith('yml'):
      f = open(fn, 'r')
      # lines = f.readlines()
      info = yaml.safe_load(f)   
      f.close()
    elif fn.endswith('json'):
      #json file
      f = open(fn, 'r')
      info = json.load(f)
      f.close()
    
    patches = info['patches']
    n_patch = len(patches)
    if n_patch > 0:
      n_patch_sample = len(patches[0]['grid']) // 3
    
    
    pred_data['pred_patch_points'] = torch.zeros([1, n_patch, n_patch_sample, 3], device = device)
    pred_data['pred_patch_logits'] = torch.zeros([1, n_patch, 2], device=device)
    pred_data['pred_patch_logits'][0][:,0] = 1
    pred_data['pred_patch_type'] = torch.zeros([1,n_patch, 6], device = device)
    pred_data['closed_patch_logits'] = torch.zeros([1, n_patch, 2], device = device)

    pred_data['pred_patch_with_param'] = torch.zeros([1, n_patch], device = device)
    if flag_with_param:
      pred_data['pred_patch_param'] = torch.zeros([1, n_patch, 7], device = device)
      pred_data['pred_patch_type_name'] = ['' for i in range(n_patch)]

    for i in range(n_patch):
      for j in range(n_patch_sample):
        for k in range(3):
          # pred_data['pred_patch_points'][0][i][j][k] = float(line[1 + 3 * j + k])
          pred_data['pred_patch_points'][0][i][j][k] = patches[i]['grid'][3 * j + k]
      pred_data['pred_patch_type'][0][i][patch_type_to_id(patches[i]['type'])] = 1.0
      if patches[i]['u_closed']: #ignore v_closed
        pred_data['closed_patch_logits'][0][i][0] = 0.0
        # pred_data['closed_patch_logits'][0][i][1] = 1.0

      else:
        pred_data['closed_patch_logits'][0][i][0] = 1.0
      pred_data['closed_patch_logits'][0][i][1] = 1.0 - pred_data['closed_patch_logits'][0][i][0]
      
      if flag_with_param and patches[i]['with_param'] == True:
        pred_data['pred_patch_with_param'][0][i]  = 1.0
        # pred_data['pred_patch_type_name'][i] = patches[i]['type']
        #detailed parameters to be add
        # print('type name: ', patches[i]['type'])
        for j in range(7):
          pred_data['pred_patch_param'][0][i][j] = patches[i]['param'][j]
    #patch2patch info to be evaluated
    #no curve info given yet
    if with_curve_corner:
      if "corners" in info:
        if info["corners"] == None:
          n_corner = 0
        else:
          n_corner = len(info["corners"])
      else:
        n_corner = 0
      pred_data['pred_corner_position'] = torch.zeros([1, n_corner, 3], device = device)
      pred_data['pred_logits'] = torch.zeros([1, n_corner, 2], device = device)
      pred_data['pred_logits'][0][:,0] = 1.0
      for i in range(n_corner):
        for j in range(3):
          pred_data['pred_corner_position'][0][i][j] = info['corners'][i]['pts'][j]
      
      n_curve = 0
      if info['curves'] != None:
          n_curve = len(info['curves'])
      pred_data['pred_curve_logits'] = torch.zeros([1, n_curve, 2], device = device)
      pred_data['pred_curve_logits'][0][:,0] = 1
      #type not set yet
      pred_data['pred_curve_type'] = torch.zeros([1, n_curve, 4], device = device) 
      
      pred_data['closed_curve_logits'] = torch.zeros([1, n_curve, 2], device = device) #not set yet
      pred_data['pred_curve_points'] = torch.zeros([1,n_curve, n_curve_sample, 3], device =device)
      for i in range(n_curve):
        for j in range(n_curve_sample):
          for k in range(3):
            # pred_data['pred_curve_points'][0][i][j][k] = float(line[2 + 3 * j + k])
            pred_data['pred_curve_points'][0][i][j][k] = info['curves'][i]['pts'][3 * j + k]
        pred_data['pred_curve_type'][0][i][curve_type_to_id(info['curves'][i]['type'])] = 1.0
        if info['curves'][i]['closed']:
          # pred_data['closed_curve_logits'][0][i][0] = 1.0
          pred_data['closed_curve_logits'][0][i][1] = 1.0

        else:
          # pred_data['closed_curve_logits'][0][i][1] = 1.0
          pred_data['closed_curve_logits'][0][i][0] = 1.0

      pred_data['patch2curve'] = info['patch2curve']
      pred_data['curve2corner'] = info['curve2corner']
      pred_data['patch2corner'] = info['patch2corner'] #added 1224
    
    return pred_data

  disable_aux_loss_output = True
  model_shape.eval()
  corner_loss_criterion.eval()
  curve_loss_criterion.eval()
  patch_loss_criterion.eval()
  assert(args.batch_size == 1)
  #test_data = train_data_loader(1, voxel_dim=voxel_dim, device=device, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, data_folder="test_data")
  data_loader_iterator = iter(train_data)
  
  # obj_dir = os.path.join("experiments", args.experiment_name, "test_obj")
  # obj_dir = os.path.join("experiments", args.experiment_name, "twomodel")
  # obj_dir = os.path.join("experiments", args.experiment_name, "exp8_fitting_1208")
  # obj_dir = os.path.join("experiments", args.experiment_name, "exp8_1_fitting")
  # obj_dir = os.path.join("experiments", args.experiment_name, "exp_81_mix_val03_simpl")
  # obj_dir = os.path.join("experiments", args.experiment_name, "exp_81_test_obj_250k_mix_val03")
  # obj_dir = os.path.join("experiments", args.experiment_name, "exp16_input128_fitting1226")
  # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp9_fitting")
  # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp6_fitting")
  # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp14_fitting")
  # obj_dir = os.path.join("experiments", args.experiment_name, "exp8_first_round_fail")

  # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp13_rest_fitting")
  # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp7_all_0106")
  # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp8_fitting_0106")

  # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp13_all_0107")
  # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp7_allnb")
  # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp7_allnb")
  # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp7_all_nb03")
  # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp7_allnb_0113")
  # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp7_600_add_fitting")
  # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp7_600_mul_fitting")
  # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp7_prediction_complex")
  # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp7_test0119_fitting")
  obj_dir = os.path.join("experiments", args.experiment_name, args.folder)









  if args.vis_test:
    # obj_dir = os.path.join("experiments", args.experiment_name, "vis_test")
    # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp9_vistest_fitting")
    # obj_dir = os.path.join("experiments", args.experiment_name, "1208_exp8_fitting_0106")
    obj_dir = os.path.join("experiments", args.experiment_name, args.folder)
  if args.vis_train:
    obj_dir = os.path.join("experiments", args.experiment_name, "vis_train")
  if(not os.path.exists(obj_dir)): os.mkdir(obj_dir)
  
  test_statistics = []
  sample_name_list = []
  
  # sample_count = 0
  def export_curves(curve_points, points_number_per_curve, output_obj_file):
        curve_points = np.reshape(curve_points, [-1,3])
        with open(output_obj_file, "w") as wf:
          for point in curve_points:
            wf.write("v {} {} {}\n".format(point[0], point[1], point[2]))
          for i in range(curve_points.shape[0]):
            if(i % points_number_per_curve == (points_number_per_curve - 1)):
              continue
            wf.write("l {} {}\n".format(i+1, i+2))

  dict_sum = {}

  #valid data from test dictionary
  path = obj_dir
  allfs = os.listdir(obj_dir)
  all_valid_id = []
  for f in allfs:
    if f.endswith(suffix):
      all_valid_id.append(f.replace(suffix, ""))
  
  all_valid_id_set = set(all_valid_id)
  
  nopatchcount = 0
  while(True):
    try:
      data_item = next(data_loader_iterator)
    except StopIteration:
      #data_loader_iterator = iter(train_data)
      #data_item = next(data_loader_iterator)
      break
    locations = data_item[0].to(device)
    features = data_item[1].to(device)
    corner_points = data_item[2].to(device)
    corner_batch_idx = data_item[3].to(device)
    batch_sample_id = data_item[5]
    target_curves_list = data_item[6]
    target_patches_list = data_item[7]
    input_pointcloud = data_item[4][0]

    sample_id = batch_sample_id[0].replace("_fix.pkl" ,"")
    print ('sample_id: ', sample_id)
    if sample_id not in all_valid_id_set:
      continue
    cur_id = sample_id
    output_pkl_name = os.path.join(path, "{}_stats.pkl".format(cur_id))
    if not args.regen and os.path.exists(output_pkl_name):
      f = open(output_pkl_name, 'rb')
      summary_loss_dict = pickle.load(f)
      # if (len(summary_loss_dict) != 42):
      #   print('summary_loss_dict: ')
      assert(len(summary_loss_dict) == 42)
      test_statistics.append(list(summary_loss_dict.values()))
      # sample_id.append(cur_id)
      sample_name_list.append(cur_id)
      f.close()

      for k in summary_loss_dict:
        if k in dict_sum:
          dict_sum[k] += summary_loss_dict[k]
        else:
          dict_sum[k] = summary_loss_dict[k]
      continue

    # sample_count+=1
    #convert target_curves_list and target_patches_list to cuda tensors
    for i in range(len(target_curves_list)):
      for k in target_curves_list[i]:
        target_curves_list[i][k] = target_curves_list[i][k].to(device)
    
    for i in range(len(target_patches_list)):
      for k in target_patches_list[i]:
        if k == 'patch_points' or k == 'patch_normals' or k == 'patch_pcs':
          for j in range(len(target_patches_list[i][k])): #list
            target_patches_list[i][k][j] = target_patches_list[i][k][j].to(device)
        else:
          # print('k: ', k)
          target_patches_list[i][k] = target_patches_list[i][k].to(device)
    
    #supervision
    batch_corner_numbers = []
    for i in range(args.batch_size):
      batch_corner_numbers.append((corner_batch_idx==i).sum())
    target_corner_points_list = torch.split(corner_points, batch_corner_numbers)
    
    #batch size should be one
    yaml_file = os.path.join(path, "{}{}".format(cur_id, suffix))
    if yaml_file.endswith('.complex'):
        pred_data = load_complex_file(yaml_file, device)
    else:
        pred_data = load_yaml_json_file(yaml_file)

    summary_loss_dict = {}
    
    if with_curve_corner:
      #corner matching
      corner_loss_dict, corner_matching_indices = corner_loss_criterion(pred_data, target_corner_points_list)
      curve_loss_dict, curve_matching_indices = curve_loss_criterion(pred_data, target_curves_list)

      #get topo information
      for k in corner_loss_dict.keys():
        summary_loss_dict["corner_" + k] = corner_loss_dict[k].item()
      
      for k in curve_loss_dict.keys():
        summary_loss_dict["curve_" + k] = curve_loss_dict[k].item()
      
      target_curves_list[0]['curve_length_weighting'][:] = 1.0

    patch_loss_dict, patch_matching_indices = patch_loss_criterion(pred_data, target_patches_list)
    for k in patch_loss_dict.keys():
        if not k == 'patch_idx_filter':
          summary_loss_dict["patch_" + k] = patch_loss_dict[k].item()
    # print('patch matching indices: ', patch_matching_indices)
    summary_loss_dict["patch_recall_filtered"] = summary_loss_dict["patch_recall"] * summary_loss_dict["patch_n_patch_filter"] / (summary_loss_dict["patch_n_patch"] + 1e-8)

    if args.eval_res_cov:
      patch_close_logits = pred_data['closed_patch_logits'][0].detach().cpu().numpy()
      patch_uclosed = patch_close_logits[:,0] < patch_close_logits[:,1]
      # distances = compute_overall_singlecd(pred_data['pred_patch_points'][0].detach().cpu().numpy(), patch_uclosed, input_pointcloud)
      if not args.eval_param:
        if args.eval_matched:
          distances, distances_matched = compute_overall_singlecd(pred_data['pred_patch_points'][0].detach().cpu().numpy(), patch_uclosed, input_pointcloud, patch_matching_indices)
        else:
          distances = compute_overall_singlecd(pred_data['pred_patch_points'][0].detach().cpu().numpy(), patch_uclosed, input_pointcloud, patch_matching_indices)
      else:
        if args.eval_matched:
          # distances, distances_matched = compute_overall_singlecd_param(pred_data, input_pointcloud, patch_matching_indices)
          distances, distances_matched = compute_overall_singlecd_param(pred_data, input_pointcloud, patch_loss_dict['patch_idx_filter'])
        else:
          distances= compute_overall_singlecd_param(pred_data, input_pointcloud, patch_matching_indices)
      
      # summary_loss_dict["overall_single_cd"] = distances.mean()
      summary_loss_dict['p_cov_1'] = (distances < 0.01).sum() / input_pointcloud.shape[0]
      if args.noisetest:
        noise_map = {1: 0.01, 2: 0.02, 3: 0.05}
        cur_noise = noise_map[args.noisetest]
        summary_loss_dict['p_cov_{}'.format(cur_noise)] = (distances < cur_noise).sum() / input_pointcloud.shape[0]
        summary_loss_dict['p_cov_2x{}'.format(cur_noise)] = (distances < 2 * cur_noise).sum() / input_pointcloud.shape[0]
      if args.eval_matched:
        summary_loss_dict['p_cov_1_matched'] = (distances_matched < 0.01).sum() / input_pointcloud.shape[0]
      # summary_loss_dict['p_cov_2'] = (distances < 0.02).sum() / input_pointcloud.shape[0]
    
    #topo error

    if with_curve_corner:
      pred_patch2curve = pred_data['patch2curve'] #lists of lists
      pred_curve2corner = pred_data['curve2corner'] #lists of lists
      pred_patch2corner = pred_data['patch2corner']
      num_gt_patches = len(target_patches_list[0]['u_closed'])
      num_gt_curves = len(target_curves_list[0]['is_closed'])
      num_gt_corners = len(corner_points)
      # print('corner pts num: ', num_gt_corners)
      if patch_matching_indices['indices'][0][0].shape[0] and curve_matching_indices['indices'][0][0].shape[0]:
        #patch curve correspondence

        pred_patch2curve = torch.tensor(pred_patch2curve, device = device, dtype = torch.float).view(len(pred_patch2curve), -1)

        pred_patch2curve = pred_patch2curve[patch_matching_indices['indices'][0][0],][:,curve_matching_indices['indices'][0][0]]
        
        gt_patch2curve = target_patches_list[0]['patch_curve_correspondence'][patch_matching_indices['indices'][0][1],][:,curve_matching_indices['indices'][0][1]]
        assert(pred_patch2curve.shape == gt_patch2curve.shape)
        # summary_loss_dict['topo_patch_curve'] = F.binary_cross_entropy(pred_patch2curve.view(-1), gt_patch2curve.view(-1)).item()
        # summary_loss_dict['topo_patch_curve'] = (pred_patch2curve - gt_patch2curve).abs().mean().item()
        
        #update 1224, loss computed based on ground truth
        summary_loss_dict['topo_patch_curve'] = ((pred_patch2curve - gt_patch2curve).abs().sum().item() + num_gt_patches * num_gt_curves - pred_patch2curve.shape[0] * pred_patch2curve.shape[1] ) / (num_gt_patches * num_gt_curves)


        #here to compute patch-patch connection
        pred_patch2patch = torch.mm(pred_patch2curve, torch.transpose(pred_patch2curve, 0, 1))
        gt_patch2patch = torch.mm(gt_patch2curve, torch.transpose(gt_patch2curve, 0, 1))
        tmpid = torch.arange(patch_matching_indices['indices'][0][0].shape[0], device=pred_patch2patch.device)
        pred_patch2patch[tmpid, tmpid] = 0 #diagonal
        gt_patch2patch[tmpid, tmpid] = 0
        pred_patch2patch[pred_patch2patch > 0.5] = 1
        gt_patch2patch[gt_patch2patch > 0.5] = 1
        
        # print("pred p2p: \n", pred_patch2patch)
        # print("gt p2p: \n", gt_patch2patch)

        # print("pred p2c: \n", pred_patch2curve)
        # print("gt p2c: \n", gt_patch2curve)

        # summary_loss_dict['topo_patch_patch'] = (pred_patch2patch - gt_patch2patch).abs().mean().item()
        # modified 1223
        num_match_patches = pred_patch2patch.shape[0]
        # print('match num: {} all num: {}'.format(num_match_patches, num_gt_patches))
        # print('curve gt num: ', num_gt_curves)

        summary_loss_dict['topo_patch_patch'] = ((pred_patch2patch - gt_patch2patch).abs().sum().item() + num_gt_patches * num_gt_patches - num_match_patches * num_match_patches) / (num_gt_patches * num_gt_patches)

        #filtered version
        # print('!!!!!!filter set len: ', len(filter_pred_id))

        filtered_idx_set = patch_loss_dict['patch_idx_filter']
        filter_pred_id = []
        # filter_gt_id = []
        for i in range(patch_matching_indices['indices'][0][0].shape[0]):
            if patch_matching_indices['indices'][0][0][i] in filtered_idx_set:
                # filter_pred_id.append(patch_matching_indices['indices'][0][0][i])
                filter_pred_id.append(i)
                # filter_gt_id.append(patch_matching_indices['indices'][0][1][i])
        # print('len id: ', len(filter_pred_id))
        pred_patch2patch_filter = pred_patch2patch[filter_pred_id][:, filter_pred_id]
        gt_patch2patch_filter = gt_patch2patch[filter_pred_id][:, filter_pred_id]
        summary_loss_dict['topo_patch_patch_filter'] = ((pred_patch2patch_filter - gt_patch2patch_filter).abs().sum().item() + num_gt_patches * num_gt_patches - len(filter_pred_id) * len(filter_pred_id)) / (num_gt_patches * num_gt_patches)


      else:
        #to be changed, based on gt size
        # num_gt_patches = len(target_patches_list[0]['u_closed'])
        # num_gt_curves = len(target_curves_list[0]['is_closed'])
        if num_gt_patches == 0 or num_gt_curves == 0:
          summary_loss_dict['topo_patch_patch'] = 0.0
          summary_loss_dict['topo_patch_patch_filter'] = 0.0
          summary_loss_dict['topo_patch_curve'] = 0.0
        else:
          summary_loss_dict['topo_patch_patch'] = 1.0
          summary_loss_dict['topo_patch_patch_filter'] = 1.0
          summary_loss_dict['topo_patch_curve'] = 1.0    
      
      if curve_matching_indices['indices'][0][0].shape[0] and corner_matching_indices['indices'][0][0].shape[0]:
        pred_curve2corner = torch.tensor(pred_curve2corner, device=device, dtype = torch.float).view(len(pred_curve2corner), -1)
        pred_curve2corner = pred_curve2corner[curve_matching_indices['indices'][0][0],][:,corner_matching_indices['indices'][0][0]]
        
        cur_curves_gt = target_curves_list[0]
        
        open_curve_idx = torch.where(cur_curves_gt['is_closed'][curve_matching_indices['indices'][0][1]] < 0.5)

        curve2corner_gt = target_curves_list[0]['endpoints'][curve_matching_indices['indices'][0][1]]
        # curve2corner_gt = target_curves_list[0]['endpoints']

        gt_curve2corner = torch.zeros([curve_matching_indices['indices'][0][1].shape[0], target_corner_points_list[0].shape[0]], device=device, dtype = torch.float)
        
        # print('gt curve2corner : ', gt_curve2corner.shape)
        # print('curve2corner: ', curve2corner_gt)
        gt_curve2corner[torch.arange(gt_curve2corner.shape[0]), curve2corner_gt[:,0]] = 1
        gt_curve2corner[torch.arange(gt_curve2corner.shape[0]), curve2corner_gt[:,1]] = 1

        for i in range(curve_matching_indices['indices'][0][1].shape[0]):
          ori_curveid = curve_matching_indices['indices'][0][1][i]
          if (target_curves_list[0]['is_closed'][ori_curveid] >= 0.5):
            gt_curve2corner[i] = 0.0

        gt_curve2corner = gt_curve2corner[:, corner_matching_indices['indices'][0][1]]
        gt_curve2corner = gt_curve2corner[open_curve_idx]
        pred_curve2corner = pred_curve2corner[open_curve_idx]
        assert(pred_curve2corner.shape == gt_curve2corner.shape)
        # summary_loss_dict['topo_curve_corner'] = F.binary_cross_entropy(pred_curve2corner.view(-1), gt_curve2corner.view(-1)).item()
        # summary_loss_dict['topo_curve_corner'] = (pred_curve2corner - gt_curve2corner).abs().mean().item()
        summary_loss_dict['topo_curve_corner'] = ((pred_curve2corner - gt_curve2corner).abs().sum().item() + num_gt_curves * num_gt_corners - pred_curve2corner.shape[0] * pred_curve2corner.shape[1]) / (num_gt_curves * num_gt_corners)
      else:
        if num_gt_curves == 0 or num_gt_corners == 0:
          summary_loss_dict['topo_curve_corner'] = 0.0
        else:
          summary_loss_dict['topo_curve_corner'] = 1.0 #with gt, without prediction

      if patch_matching_indices['indices'][0][0].shape[0] and corner_matching_indices['indices'][0][0].shape[0] and curve_matching_indices['indices'][0][0].shape[0]:
        pred_patch2corner = torch.tensor(pred_patch2corner, device=device, dtype = torch.float).view(len(pred_patch2corner), -1)
        pred_patch2corner = pred_patch2corner[patch_matching_indices['indices'][0][0],][:,corner_matching_indices['indices'][0][0]]

        cur_curves_gt = target_curves_list[0]
        
        open_curve_idx = torch.where(cur_curves_gt['is_closed'][curve_matching_indices['indices'][0][1]] < 0.5)

        curve2corner_gt = target_curves_list[0]['endpoints'][curve_matching_indices['indices'][0][1]]
        # curve2corner_gt = target_curves_list[0]['endpoints']

        gt_curve2corner = torch.zeros([curve_matching_indices['indices'][0][1].shape[0], target_corner_points_list[0].shape[0]], device=device, dtype = torch.float)
        
        # print('gt curve2corner : ', gt_curve2corner.shape)
        # print('curve2corner: ', curve2corner_gt)
        gt_curve2corner[torch.arange(gt_curve2corner.shape[0]), curve2corner_gt[:,0]] = 1
        gt_curve2corner[torch.arange(gt_curve2corner.shape[0]), curve2corner_gt[:,1]] = 1

        for i in range(curve_matching_indices['indices'][0][1].shape[0]):
          ori_curveid = curve_matching_indices['indices'][0][1][i]
          if (target_curves_list[0]['is_closed'][ori_curveid] >= 0.5):
            gt_curve2corner[i] = 0.0

        gt_curve2corner = gt_curve2corner[:, corner_matching_indices['indices'][0][1]]
        gt_curve2corner = gt_curve2corner[open_curve_idx]

        gt_patch2curve = target_patches_list[0]['patch_curve_correspondence'][patch_matching_indices['indices'][0][1],][:,curve_matching_indices['indices'][0][1]]

        
        gt_patch2corner = torch.mm(gt_patch2curve[:, open_curve_idx[0]], gt_curve2corner)
        
        gt_patch2corner[gt_patch2corner > 1.0] = 1.0
        assert(gt_patch2corner.shape == pred_patch2corner.shape)
        summary_loss_dict['topo_patch_corner'] = ((pred_patch2corner - gt_patch2corner).abs().sum().item() + num_gt_patches * num_gt_corners - pred_patch2corner.shape[0] * pred_patch2corner.shape[1]) / (num_gt_patches * num_gt_corners)
      else:
        if num_gt_patches == 0 or num_gt_corners == 0 or num_gt_curves == 0:
          summary_loss_dict['topo_patch_corner'] = 0.0
        else:
          summary_loss_dict['topo_patch_corner'] = 1.0

      if args.eval_selftopo:
        #self topo consistency
        pred_patch2curve = pred_data['patch2curve'] #lists of lists
        pred_curve2corner = pred_data['curve2corner'] #lists of lists
        pred_patch2corner = pred_data['patch2corner']
        #curve not none
        summary_loss_dict['cons_curve_point'] = 0.0
        summary_loss_dict['cons_curve_patch'] = 0.0
        summary_loss_dict['cons_matrix'] = 0.0
        
        if not pred_patch2curve == None and len(pred_patch2curve[0]) > 0:
          #patch curve not none
          pred_patch2curve = torch.tensor(pred_patch2curve, device = device, dtype = torch.float).view(len(pred_patch2curve), -1)
          curve_patch_cons = (torch.sum(pred_patch2curve, dim=0) - 2).abs().mean().item()
          summary_loss_dict['cons_curve_patch'] = curve_patch_cons
          if not pred_patch2corner == None and len(pred_patch2corner[0]) > 0:
            #corner not none
            pred_patch2corner = torch.tensor(pred_patch2corner, device=device, dtype = torch.float).view(len(pred_patch2corner), -1)
            if len(pred_curve2corner) > 0:
                pred_curve2corner = torch.tensor(pred_curve2corner, device=device, dtype = torch.float).view(len(pred_curve2corner), -1)
                #pred open idx to get
                # np.savetxt('closed_logit.txt', pred_data['closed_curve_logits'][0].cpu().numpy())
                open_curve_idx = torch.where(pred_data['closed_curve_logits'][0][:, 0].cpu()> pred_data['closed_curve_logits'][0][:, 1].cpu())
                # print('all curve size: {} open curve size: {}'.format(pred_data['closed_curve_logits'][0].shape, open_curve_idx[0].shape))
                if len(open_curve_idx[0]) > 0:
                    curve_point_cons = (torch.sum(pred_curve2corner[open_curve_idx[0]], dim=1) - 2).abs().mean().item()
                    summary_loss_dict['cons_curve_point'] = curve_point_cons
                # curve_point_cons = (torch.sum(pred_curve2corner, dim=1) - 2).abs().sum().item()
                
                # print('mc shape {} cc shape {} patch corner shape {}'.format(pred_patch2curve.shape, pred_curve2corner.shape, pred_patch2corner.shape))
                matrix_cons = (torch.mm(pred_patch2curve, pred_curve2corner) - 2.0 * pred_patch2corner).abs().mean().item()
                summary_loss_dict['cons_matrix'] = matrix_cons
        
      #udpate 0107, eval topo-geom consistency
      pred_patch2curve = pred_data['patch2curve'] #lists of lists
      pred_curve2corner = pred_data['curve2corner'] #lists of lists
      pred_patch2corner = pred_data['patch2corner']

      dist_topo_geom = 0.0
      flag_mean = True
      
      #update 0113, mean version
      #update 0118, add valid ratio
      pair_dists = []
      topo_sum = 0

      patch_pos = pred_data['pred_patch_points'][0]
      curve_pos = pred_data['pred_curve_points'][0]
      corner_pos = pred_data['pred_corner_position'][0]
      # print('patch shape: {} curve: {} corner: {}'.format(patch_pos.shape, curve_pos.shape, corner_pos.shape))
      assert(len(pred_patch2curve) == len(patch_pos))
      # assert(len(pred_curve2corner) == len(curve_pos))
      if len(patch_pos) > 0 and len(curve_pos) > 0:
        #patch2curve
        # tmp_cd_all = []
        pred_patch2curve = torch.tensor(pred_patch2curve, device = device, dtype = torch.int64).view(len(pred_patch2curve), -1)
        topo_sum += pred_patch2curve.sum()
        for k in range(len(patch_pos)):
          if pred_patch2curve[k].sum() > 0:
            nn_curve = curve_pos[pred_patch2curve[k] > 0.5]
            # print('nn curve shape: ', nn_curve.shape)
            nn_curve = nn_curve.view(1,-1,3)
            cur_patch = patch_pos[k].view(1,-1,3)
            target_nn = knn_points(nn_curve, cur_patch)
            cur_cd = target_nn.dists[...,0].sqrt()
            # target_id = target_nn.idx
            # print('patch curve cd shape: ', cur_cd.shape)

            if not flag_mean:
              if cur_cd.max() > args.dist_th_tg:
                print('max dist: ', cur_cd.max())
                print('curve inconsistency for patch: {}\n'.format(k))
              if dist_topo_geom < cur_cd.max():
                dist_topo_geom = cur_cd.max().item()
            else:
              if cur_cd.mean() > args.dist_th_tg:
                print('mean dist: ', cur_cd.mean())
                print('curve inconsistency for patch: {}\n'.format(k))
              if dist_topo_geom < cur_cd.mean():
                dist_topo_geom = cur_cd.mean().item()

            cur_cd = cur_cd.view(-1, 34)
            for i in range(len(cur_cd)):
                pair_dists.append(cur_cd[i].mean().item())

      
      if len(patch_pos) > 0 and len(corner_pos) > 0:
        pred_patch2corner = torch.tensor(pred_patch2corner, device = device, dtype = torch.int64).view(len(pred_patch2corner), -1)
        topo_sum += pred_patch2corner.sum()
        for k in range(len(patch_pos)):
          if pred_patch2corner[k].sum() > 0:
            nn_corner = corner_pos[pred_patch2corner[k] > 0.5]
            # print('nn curve shape: ', nn_curve.shape)
            nn_corner = nn_corner.view(1,-1,3)
            cur_patch = patch_pos[k].view(1,-1,3)
            target_nn = knn_points(nn_corner, cur_patch)
            cur_cd = target_nn.dists[...,0].sqrt()
            # target_id = target_nn.idx
            # print('patch corner cd shape: ', cur_cd.shape)

            
            if not flag_mean:
              if cur_cd.max() > args.dist_th_tg:
                print('max dist: ', cur_cd.max())
                print('corner inconsistency for patch: {}\n'.format(k))
              if dist_topo_geom < cur_cd.max():
                dist_topo_geom = cur_cd.max().item()
            else:
              if cur_cd.mean() > args.dist_th_tg:
                print('mean dist: ', cur_cd.mean())
                print('corner inconsistency for patch: {}\n'.format(k))
              if dist_topo_geom < cur_cd.mean():
                dist_topo_geom = cur_cd.mean().item()

            cur_cd = cur_cd.view(-1)
            for i in range(len(cur_cd)):
                pair_dists.append(cur_cd[i].item())
      
      if len(curve_pos) > 0 and len(corner_pos) > 0:
        pred_curve2corner = torch.tensor(pred_curve2corner, device = device, dtype = torch.int64).view(len(pred_curve2corner), -1)
        topo_sum += pred_curve2corner.sum()
        for k in range(len(curve_pos)):
          if pred_curve2corner[k].sum() > 0:
            nn_corner = corner_pos[pred_curve2corner[k] > 0.5]
            # print('nn curve shape: ', nn_curve.shape)
            nn_corner = nn_corner.view(1,-1,3)
            cur_curve = curve_pos[k].view(1,-1,3)
            target_nn = knn_points(nn_corner, cur_curve)
            cur_cd = target_nn.dists[...,0].sqrt()
            # target_id = target_nn.idx
            # print('curve corner cd shape: ', cur_cd.shape)
            if not flag_mean:
              if cur_cd.max() > args.dist_th_tg:
                print('max dist: ', cur_cd.max())
                print('corner inconsistency for curve: {}\n'.format(k))
              if dist_topo_geom < cur_cd.max():
                dist_topo_geom = cur_cd.max().item()
            else:
              if cur_cd.mean() > args.dist_th_tg:
                print('mean dist: ', cur_cd.mean())
                print('corner inconsistency for curve: {}\n'.format(k))
              if dist_topo_geom < cur_cd.mean():
                dist_topo_geom = cur_cd.mean().item()

            cur_cd = cur_cd.view(-1)
            for i in range(len(cur_cd)):
                pair_dists.append(cur_cd[i].item())
        
      
      summary_loss_dict['topo_geom_dist'] = dist_topo_geom
      summary_loss_dict['topo_geom_cons_01'] = (dist_topo_geom < 0.1) * 1
      summary_loss_dict['topo_geom_cons_02'] = (dist_topo_geom < 0.2) * 1

      pair_dists = np.array(pair_dists)
      summary_loss_dict['topo_valid_ratio01'] = 1.0
      summary_loss_dict['topo_valid_ratio005'] = 1.0
      summary_loss_dict['topo_valid_ratio004'] = 1.0
      summary_loss_dict['topo_valid_ratio003'] = 1.0
      summary_loss_dict['topo_valid_ratio002'] = 1.0
      summary_loss_dict['topo_valid_ratio001'] = 1.0
      # print('pair dist shape: ', pair_dists.shape)
      # print('topo sum: ', topo_sum)
      if len(pair_dists) > 0:
        summary_loss_dict['topo_valid_ratio01'] = (pair_dists < 0.1).sum() / len(pair_dists)
        summary_loss_dict['topo_valid_ratio005'] = (pair_dists < 0.05).sum() / len(pair_dists)
        summary_loss_dict['topo_valid_ratio004'] = (pair_dists < 0.04).sum() / len(pair_dists)
        summary_loss_dict['topo_valid_ratio003'] = (pair_dists < 0.03).sum() / len(pair_dists)
        summary_loss_dict['topo_valid_ratio002'] = (pair_dists < 0.02).sum() / len(pair_dists)
        summary_loss_dict['topo_valid_ratio001'] = (pair_dists < 0.01).sum() / len(pair_dists)
      

    # print('length of summaray loss dict: ', len(summary_loss_dict))
    # print('summaray loss dict: ', summary_loss_dict)

    for k in summary_loss_dict:
      if k in dict_sum:
        dict_sum[k] += summary_loss_dict[k]
      else:
        dict_sum[k] = summary_loss_dict[k]

    # print(list(summary_loss_dict.values()))
    # assert(len(summary_loss_dict) == 13)
    test_statistics.append(list(summary_loss_dict.values()))
    # sample_id.append(cur_id)
    sample_name_list.append(cur_id)
    f = open(output_pkl_name,"wb")
    pickle.dump(summary_loss_dict, f)
    f.close()

  for k in dict_sum:
    dict_sum[k] = dict_sum[k] / (len(sample_name_list))

  print('dict sum: ', dict_sum)
  sample_name_list.append("mean")
  test_statistics.append(list(dict_sum.values()))

  # obj
  # comment on 1223
  np.savetxt(os.path.join(obj_dir, "final_evaluation" + suffix + ".txt"), np.array(test_statistics))
  with open(os.path.join(obj_dir, "test_final_sample_name.txt"), "w") as wf:
    for sample_name in sample_name_list:
      wf.write("{}\n".format(sample_name))
  print(len(sample_name_list) - 1, "samples in test set")
  
  import pandas as pd
  ## convert your array into a dataframe
  df = pd.DataFrame(np.array(test_statistics))
  # df.columns = list(summary_loss_dict.keys())
  df.columns = list(dict_sum.keys())
  df.index = sample_name_list
  #title_row = pd.Series()
  #row_df = pd.DataFrame([title_row])
  #df = pd.concat([row_df, df], ignore_index=True)
  
  ## save to xlsx file
  filepath = os.path.join(obj_dir, 'final_evaluation' + suffix + '.xlsx')
  if args.part >= 0:
    filepath = filepath.replace('.xlsx', '_part{}.xlsx'.format(args.part))
  
  df.to_excel(filepath, index=True)




def get_val_summary_dict(model_shape, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion, train_data, device, train_iter, summary_ref):
  #infact , val_data is used here
  disable_aux_loss_output = True
  model_shape.eval()
  corner_loss_criterion.eval()
  curve_loss_criterion.eval()
  patch_loss_criterion.eval()

  # model_shape.train()
  # corner_loss_criterion.train()
  # curve_loss_criterion.train()
  # patch_loss_criterion.train()

  # assert(args.batch_size == 1)
  #test_data = train_data_loader(1, voxel_dim=voxel_dim, device=device, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, data_folder="test_data")
  data_loader_iterator = iter(train_data)
  
  test_statistics = []
  # sample_name_list = []
  sample_count = 0
  summary_loss_dict = {}
  for k in summary_ref:
    summary_loss_dict[k] = torch.zeros(1, device = device)

  while(True):
    try:
      data_item = next(data_loader_iterator)
    except StopIteration:
      #data_loader_iterator = iter(train_data)
      #data_item = next(data_loader_iterator)
      break
    sample_count+=1
    # print ("sample count: ", sample_count)
    locations = data_item[0].to(device)
    features = data_item[1].to(device)
    corner_points = data_item[2].to(device)
    corner_batch_idx = data_item[3].to(device)
    batch_sample_id = data_item[5]
    target_curves_list = data_item[6]
    target_patches_list = data_item[7]

    #convert target_curves_list and target_patches_list to cuda tensors
    for i in range(len(target_curves_list)):
      for k in target_curves_list[i]:
        target_curves_list[i][k] = target_curves_list[i][k].to(device)
    
    for i in range(len(target_patches_list)):
      for k in target_patches_list[i]:
        if k == 'patch_points' or k == 'patch_normals' or k == 'patch_pcs':
          for j in range(len(target_patches_list[i][k])): #list
            target_patches_list[i][k][j] = target_patches_list[i][k][j].to(device)
        else:
          target_patches_list[i][k] = target_patches_list[i][k].to(device)

    
    #supervision
    batch_corner_numbers = []
    for i in range(args.batch_size):
      batch_corner_numbers.append((corner_batch_idx==i).sum())
    target_corner_points_list = torch.split(corner_points, batch_corner_numbers)
    
    #forward
    sparse_locations, corner_predictions, curve_predictions, patch_predictions = model_shape(locations, features)
    input_pointcloud = data_item[4][0]
    
    #curves
    labels = torch.argmax(curve_predictions['pred_curve_logits'].softmax(-1)[0], dim=-1).cpu().numpy()
    #curve_type = torch.argmax(curve_predictions['pred_curve_type'].softmax(-1)[0], dim=-1).cpu().numpy()
    
    pred_curve_type = curve_type_list[torch.argmax(curve_predictions['pred_curve_type'].softmax(-1)[0], dim=-1).cpu().numpy().astype(np.int32)[np.where(labels == 0)]].tolist()
    gt_curve_type = curve_type_list[target_curves_list[0]['labels'].cpu().numpy().astype(np.int32)].tolist()
    
    curve_points = curve_predictions['pred_curve_points'][0].detach().cpu().numpy()
    effective_curve_points = np.reshape(curve_points[np.where(labels == 0)], [-1,3])
    #np.savetxt(os.path.join(obj_dir, "pred_curves_{:06d}.xyz".format(sample_id)), effective_curve_points)
    
    empty_curve_points = np.reshape(curve_points[np.where(labels == 1)], [-1,3])
    #np.savetxt(os.path.join(obj_dir, "pred_curves_empty_{:06d}.xyz".format(sample_id)), empty_curve_points)
    
    target_curve_points = np.reshape(target_curves_list[0]['curve_points'].cpu().numpy(), [-1,3])
    #np.savetxt(os.path.join(obj_dir, "gt_curves_{:06d}.xyz".format(sample_id)), target_curve_points)
    
    #corners
    labels = torch.argmax(corner_predictions['pred_logits'].softmax(-1)[0], dim=-1).cpu().numpy()
    corner_position = corner_predictions['pred_corner_position'][0].detach().cpu().numpy()
    effective_corner_position = corner_position[np.where(labels == 0)]
    
    empty_corner_position = corner_position[np.where(labels == 1)]
    
    target_corner_position = target_corner_points_list[0].cpu().numpy()
    
    #patches
    patch_labels = torch.argmax(patch_predictions['pred_patch_logits'].softmax(-1)[0], dim=-1).cpu().numpy()
    patch_points = patch_predictions['pred_patch_points'][0].detach().cpu().numpy() #in shape [100, 100*100, 3]
    effective_patch_points = patch_points[np.where(patch_labels == 0)]
    
    pred_patch_type = patch_type_list[torch.argmax(patch_predictions['pred_patch_type'].softmax(-1)[0], dim=-1).cpu().numpy().astype(np.int32)[np.where(patch_labels == 0)]].tolist()
    gt_patch_type = patch_type_list[target_patches_list[0]['labels'].cpu().numpy().astype(np.int32)].tolist()
    
    max_norm = 0
    #losses
    # curve_loss_dict, curve_matching_indices = curve_loss_criterion(curve_predictions, target_curves_list)
    # curve_weight_dict = curve_loss_criterion.weight_dict
    # corner_loss_dict, corner_matching_indices = corner_loss_criterion(corner_predictions, target_corner_points_list)
    # corner_weight_dict = corner_loss_criterion.weight_dict
    # patch_loss_dict, patch_matching_indices = patch_loss_criterion(patch_predictions, target_patches_list)
    # patch_weight_dict = patch_loss_criterion.weight_dict


    #ori version
    curve_loss_dict, curve_matching_indices = curve_loss_criterion(curve_predictions, target_curves_list)
    curve_weight_dict = curve_loss_criterion.weight_dict
    corner_loss_dict, corner_matching_indices = corner_loss_criterion(corner_predictions, target_corner_points_list)
    corner_weight_dict = corner_loss_criterion.weight_dict
    patch_loss_dict, patch_matching_indices = patch_loss_criterion(patch_predictions, target_patches_list)
    patch_weight_dict = patch_loss_criterion.weight_dict

    
    #print(loss_dict)
    corner_losses = sum(corner_loss_dict[k].detach() * corner_weight_dict[k] for k in corner_loss_dict.keys() if k in corner_weight_dict)
    curve_losses = sum(curve_loss_dict[k].detach() * curve_weight_dict[k] for k in curve_loss_dict.keys() if k in curve_weight_dict)
    patch_losses = sum(patch_loss_dict[k].detach() * patch_weight_dict[k] for k in patch_loss_dict.keys() if k in patch_weight_dict)
    losses = corner_losses + curve_losses + patch_losses
        
    
    for k in corner_loss_dict.keys():
      if k in corner_weight_dict:
        if(disable_aux_loss_output and "_aux_" in k):
          continue
        summary_loss_dict["corner_" + k] += corner_weight_dict[k] * corner_loss_dict[k].detach()
    
    for k in curve_loss_dict.keys():
      if k in curve_weight_dict:
        if(disable_aux_loss_output and "_aux_" in k):
          continue
        summary_loss_dict["curve_" + k] += curve_weight_dict[k] * curve_loss_dict[k].detach()
    
    for k in patch_loss_dict.keys():
      if k in patch_weight_dict:
        if(disable_aux_loss_output and "_aux_" in k):
          continue
        summary_loss_dict["patch_" + k] += patch_weight_dict[k] * patch_loss_dict[k].detach()
    
    #accuracy of corner and curve predictions
    # summary_loss_dict['corner_valid_accuracy'] += corner_loss_dict['corner_prediction_accuracy']
    # summary_loss_dict['curve_valid_accuracy'] += curve_loss_dict['valid_class_accuracy']
    # summary_loss_dict['curve_type_accuracy'] += curve_loss_dict['type_class_accuracy']

    summary_loss_dict['corner_valid_accuracy'] += corner_loss_dict['corner_prediction_accuracy']
    summary_loss_dict['curve_valid_accuracy'] += curve_loss_dict['valid_class_accuracy']
    summary_loss_dict['curve_type_accuracy'] += curve_loss_dict['type_class_accuracy']
    summary_loss_dict['patch_valid_accuracy'] += patch_loss_dict['valid_class_accuracy']
    summary_loss_dict['patch_type_accuracy'] += patch_loss_dict['type_class_accuracy']

    #overall accuracy
    summary_loss_dict['corner_valid_accuracy_overall'] += corner_loss_dict['corner_prediction_accuracy_overall']
    summary_loss_dict['curve_valid_accuracy_overall'] += curve_loss_dict['valid_class_accuracy_overall']
    summary_loss_dict['patch_valid_accuracy_overall'] += patch_loss_dict['valid_class_accuracy_overall']
    
    if not args.no_topo:
      if(args.curve_corner_geom_loss_coef > 0 or args.curve_corner_topo_loss_coef > 0):
        if not args.topo_acc:
          if args.ori_topo:
            curve_corner_matching_loss_geom, curve_corner_matching_loss_topo, all_zero_corners = \
              Curve_Corner_Matching_v2(corner_predictions, curve_predictions, target_corner_points_list, target_curves_list, corner_matching_indices['indices'], curve_matching_indices['indices'])
          else:
            curve_corner_matching_loss_geom, curve_corner_matching_loss_topo, all_zero_corners = \
              Curve_Corner_Matching_tripath(corner_predictions, curve_predictions, target_corner_points_list, target_curves_list, corner_matching_indices['indices'], curve_matching_indices['indices'])
        else:
          curve_corner_matching_loss_geom, curve_corner_matching_loss_topo, all_zero_corners, curve_corner_topo_acc = \
            Curve_Corner_Matching_tripath(corner_predictions, curve_predictions, target_corner_points_list, target_curves_list, corner_matching_indices['indices'], curve_matching_indices['indices'])
        # if(sample_count == 1): print("with curve corner correspondence loss")
        try:
          losses += args.curve_corner_geom_loss_coef*curve_corner_matching_loss_geom.detach() + args.curve_corner_topo_loss_coef*curve_corner_matching_loss_topo.detach()
        except:
          print(args.curve_corner_geom_loss_coef*curve_corner_matching_loss_geom)
          print(args.curve_corner_topo_loss_coef*curve_corner_matching_loss_topo)
          print(losses)
          print(losses.shape)
          print((args.curve_corner_geom_loss_coef*curve_corner_matching_loss_geom).shape)
          raise Exception("Error")
        
        summary_loss_dict['corner_curve_topo'] += args.curve_corner_topo_loss_coef*curve_corner_matching_loss_topo.detach()
        summary_loss_dict['corner_curve_geom'] += args.curve_corner_geom_loss_coef*curve_corner_matching_loss_geom.detach()

        if args.topo_acc:
          summary_loss_dict['corner_curve_topoacc'] += curve_corner_topo_acc.detach()
        if('aux_outputs' in corner_predictions):
          assert('aux_outputs' in curve_predictions)
          for aux_id in range(len(corner_matching_indices['aux_outputs'])):
            curve_corner_matching_loss_geom_aux, curve_corner_matching_loss_topo_aux = \
              Curve_Corner_Matching_v2(corner_predictions['aux_outputs'][aux_id], curve_predictions['aux_outputs'][aux_id], target_corner_points_list, target_curves_list, corner_matching_indices['aux_outputs'][aux_id]['indices'], curve_matching_indices['aux_outputs'][aux_id]['indices'])
            losses += args.curve_corner_geom_loss_coef*curve_corner_matching_loss_geom_aux + args.curve_corner_topo_loss_coef*curve_corner_matching_loss_topo_aux
            #summary_loss_dict['corner_curve_topo_aux_'+str(aux_id)] = args.curve_corner_topo_loss_coef*curve_corner_matching_loss_topo_aux.cpu().detach().numpy()

      if(args.patch_curve_topo_loss_coef > 0):
        if not args.topo_acc:
          if args.ori_topo:
            patch_curve_matching_loss_topo = \
              Patch_Curve_Matching(curve_predictions, patch_predictions, target_curves_list, target_patches_list, curve_matching_indices['indices'], patch_matching_indices['indices'])
          else:
            patch_curve_matching_loss_topo = \
              Patch_Curve_Matching_tripath(curve_predictions, patch_predictions, target_curves_list, target_patches_list, curve_matching_indices['indices'], patch_matching_indices['indices'])    
        else:
          patch_curve_matching_loss_topo, patch_curve_topo_acc = \
              Patch_Curve_Matching_tripath(curve_predictions, patch_predictions, target_curves_list, target_patches_list, curve_matching_indices['indices'], patch_matching_indices['indices'])    
        losses += args.patch_curve_topo_loss_coef*patch_curve_matching_loss_topo.detach()
        summary_loss_dict['patch_curve_topo'] += args.patch_curve_topo_loss_coef*patch_curve_matching_loss_topo.detach()
        if args.topo_acc:
          summary_loss_dict['patch_curve_topoacc'] += patch_curve_topo_acc.detach()
        if('aux_outputs' in corner_predictions):
          assert('aux_outputs' in curve_predictions)
          for aux_id in range(len(corner_matching_indices['aux_outputs'])):
            patch_curve_matching_loss_topo_aux = \
              Patch_Curve_Matching(curve_predictions['aux_outputs'][aux_id], patch_predictions['aux_outputs'][aux_id], target_curves_list, target_patches_list, curve_matching_indices['aux_outputs'][aux_id]['indices'], patch_matching_indices['aux_outputs'][aux_id]['indices'])
            losses += args.patch_curve_topo_loss_coef*patch_curve_matching_loss_topo_aux

      #to be added for evaluation
      if (args.patch_corner_topo_loss_coef > 0):
        if not args.topo_acc:
          patch_corner_matching_loss_topo, curve_point_loss, curve_patch_loss, patch_close_loss = \
            Patch_Corner_Matching_tripath(corner_predictions, curve_predictions, patch_predictions, target_corner_points_list, target_curves_list, target_patches_list, corner_matching_indices['indices'],curve_matching_indices['indices'], patch_matching_indices['indices'])
        else:
          patch_corner_matching_loss_topo, curve_point_loss, curve_patch_loss, patch_close_loss, patch_corner_topo_acc = \
            Patch_Corner_Matching_tripath(corner_predictions, curve_predictions, patch_predictions, target_corner_points_list, target_curves_list, target_patches_list, corner_matching_indices['indices'],curve_matching_indices['indices'], patch_matching_indices['indices'])
        losses += args.patch_corner_topo_loss_coef*patch_corner_matching_loss_topo
        #four other term: 
        losses += args.topo_loss_coef * curve_point_loss
        losses += args.topo_loss_coef * curve_patch_loss
        losses += args.topo_loss_coef * patch_close_loss

        # if(rank >= 0):
          # if(train_iter == 0): print("with patch corner correspondence loss")
        summary_loss_dict['patch_corner_topo'] += args.patch_corner_topo_loss_coef*patch_corner_matching_loss_topo.detach()
        if args.topo_acc:
          summary_loss_dict['patch_corner_topoacc'] += patch_corner_topo_acc.detach()
        
        if not args.no_show_topo:
          summary_loss_dict['curve_point_loss'] += args.topo_loss_coef * curve_point_loss.detach()
          summary_loss_dict['curve_patch_loss'] += args.topo_loss_coef * curve_patch_loss.detach()
          summary_loss_dict['patch_close_loss'] += args.topo_loss_coef * patch_close_loss.detach()

    summary_loss_dict['total_loss'] += losses.detach()
    
    # now = datetime.now()
    # print("{} sample:{}".format(now, sample_id))
    # print(summary_loss_dict)
    
    # sample_name_list.append(sample_id)
    # test_statistics.append(list(summary_loss_dict.values()))
    
    #train_summary = tf_summary_from_dict(summary_loss_dict, True)
    #summary_writer.add_summary(tf.compat.v1.Summary(value=train_summary), train_iter)
    
    # reduce losses over all GPUs for logging purposes
    # loss_dict_reduced = reduce_dict(corner_loss_dict)
    # #loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
    # loss_dict_reduced_scaled = {k: v * corner_weight_dict[k] for k, v in loss_dict_reduced.items() if k in corner_weight_dict}
    # losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
    # loss_value = losses_reduced_scaled.item()
  
    # if not math.isfinite(loss_value):
    #     print("Corner Loss is {}, stopping training".format(loss_value))
    #     print(loss_dict_reduced)
    #     print(batch_sample_id)
    #     sys.exit(1)
    
    # loss_dict_reduced = reduce_dict(curve_loss_dict)
    # #loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
    # loss_dict_reduced_scaled = {k: v * curve_weight_dict[k] for k, v in loss_dict_reduced.items() if k in curve_weight_dict}
    # losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
    # loss_value = losses_reduced_scaled.item()
  
    # if not math.isfinite(loss_value):
    #     print("Loss is {}, stopping training".format(loss_value))
    #     print(loss_dict_reduced)
    #     print(batch_sample_id)
    #     sys.exit(1)
  
  for k in summary_loss_dict:
    summary_loss_dict[k] = summary_loss_dict[k] / sample_count
  # print(summary_loss_dict)
  return summary_loss_dict
  # print(summary_loss_dict.keys())
  # np.savetxt(os.path.join(obj_dir, "test_statistics.txt"), np.array(test_statistics))
  # with open(os.path.join(obj_dir, "test_sample_name.txt"), "w") as wf:
  #   for sample_name in sample_name_list:
  #     wf.write("{}\n".format(sample_name))
  # print(sample_count, "samples in test set")
  
  # import pandas as pd
  # ## convert your array into a dataframe
  # df = pd.DataFrame(np.array(test_statistics))
  # df.columns = list(summary_loss_dict.keys())
  # df.index = sample_name_list
  # #title_row = pd.Series()
  # #row_df = pd.DataFrame([title_row])
  # #df = pd.concat([row_df, df], ignore_index=True)
  
  # ## save to xlsx file
  # filepath = os.path.join(obj_dir, 'test_statistics.xlsx')
  # df.to_excel(filepath, index=True)

def get_val_summary_dict_backbone(encoderdecoder, val_data, device):
  encoderdecoder.eval()
  data_loader_iterator = iter(val_data)
  summary_loss_dict = {}
  summary_loss_dict['pos'] = torch.zeros(1, device = device)
  if args.input_normal_signals:
    summary_loss_dict['normal'] = torch.zeros(1, device = device)
  summary_loss_dict['one'] = torch.zeros(1, device = device)
  summary_loss_dict['total_noweight'] = torch.zeros(1, device = device)
  sample_count = 0

  while(True):
    try:
      data_item = next(data_loader_iterator)
    except StopIteration:
      #data_loader_iterator = iter(train_data)
      #data_item = next(data_loader_iterator)
      break
    sample_count+=1
    # print ("sample count: ", sample_count)
    locations = data_item[0].to(device)
    features = data_item[1].to(device)
    gt_loc = data_item[2].to(device)
    gt_fea = data_item[3].to(device)
    gt_pos = gt_loc[:,:3]
    gt_batch_idx = gt_loc[:,-1:]
    gt_loc = torch.cat([gt_batch_idx, gt_pos], dim = 1)

    # dec_loc, dec_fea = encoderdecoder(locations, features)
    output = encoderdecoder(locations, features)
    # dec_loc = output.C
    # dec_fea = output.F
    # losses = (features - dec_fea).square().sum(-1).mean()
    # visualization 
    # np.savetxt('input_fea.xyz', features[:,:6].detach().cpu().numpy())
    # np.savetxt('output_fea.xyz', dec_fea[:,:6].detach().cpu().numpy())
    # np.savetxt('input_pos.xyz', locations[:,:3].detach().cpu().numpy())
    # np.savetxt('output_pos.xyz', dec_loc[:,1:].detach().cpu().numpy())

    gt_st = ME.SparseTensor(features=gt_fea, coordinates=gt_loc, coordinate_manager = output.coordinate_manager)
    fea_diff = (gt_st - output).F
    
    #output noise
    # np.savetxt('input_fea.xyz', features[:,:6].detach().cpu().numpy())
    # np.savetxt('input_gt_fea.xyz', gt_fea[:,:6].detach().cpu().numpy())
    
    summary_loss_dict = {}
    # summary_loss_dict['pos'] = (features - dec_fea)[:,:3].square().sum(-1).mean().detach()
    summary_loss_dict['pos'] = fea_diff[:,:3].square().sum(-1).mean().detach()

    # losses = (features - dec_fea)[:,:3].square().sum(-1).mean() * w_pos
    
    if args.input_normal_signals:
      # summary_loss_dict['normal'] = (features - dec_fea)[:,3:6].square().sum(-1).mean().detach()
      summary_loss_dict['normal'] = fea_diff[:,3:6].square().sum(-1).mean().detach()

    summary_loss_dict['one'] = fea_diff[:,-1:].square().sum(-1).mean().detach()
    summary_loss_dict['total_noweight'] = fea_diff.square().sum(-1).mean().detach()

    #ori version
    
    # # dec_loc, dec_fea = encoderdecoder(locations, features)
    # output = encoderdecoder(locations, features)
    # dec_loc = output.C
    # dec_fea = output.F
    # summary_loss_dict['pos'] += (features - dec_fea)[:,:3].square().sum(-1).mean().detach()
    
    # if args.input_normal_signals:
    #   summary_loss_dict['normal'] += (features - dec_fea)[:,3:6].square().sum(-1).mean().detach()
    # summary_loss_dict['one'] += (features - dec_fea)[:,6:].square().sum(-1).mean().detach()
    # summary_loss_dict['total'] += (features - dec_fea).square().sum(-1).mean().detach()

  for k in summary_loss_dict:
    summary_loss_dict[k] = summary_loss_dict[k] / sample_count
  
  return summary_loss_dict

def export_patches(patch_points, export_obj_filename):
  patch_points_reshape = np.reshape(patch_points, [-1, 3])
  assert(patch_points_reshape.shape[0] % 100 == 0)
  num_of_valid_patches = patch_points_reshape.shape[0] // 100
  with open(export_obj_filename, "w") as wf:
    wf.write("mtllib default.mtl\n")
    num_mtl_in_lib = 15
    for patch_idx in range(num_of_valid_patches):
      wf.write("usemtl m{}\n".format(patch_idx % num_mtl_in_lib))
      for point in patch_points_reshape[100*patch_idx:100*patch_idx+100]:
        wf.write("v %lf %lf %lf\n" %(point[0], point[1], point[2]))
      face_displacement = patch_idx*100 + 1
      for row_idx in range(9):
        for col_idx in range(10):
          vert_idx = face_displacement + row_idx*10 + col_idx
          if(col_idx != 0):
            wf.write("f %d %d %d\n" %(vert_idx, vert_idx-1, vert_idx+10))
          if(col_idx != 9):
            wf.write("f %d %d %d\n" %(vert_idx, vert_idx+10, vert_idx+11))  

def generate_random_color_palette(n_color, flag_partnet = False):
  if flag_partnet:
    np.random.seed(1)
  else:
    np.random.seed(0)
  return np.random.rand(n_color, 3)

def save_mesh_off(filename, vertices, faces, f_mask):
  if not f_mask.shape[0] == 0:
    n_color = np.max(f_mask) + 1
  else:
    n_color = 1
  print('n color: ', n_color)
  colormap = np.round(255 * generate_random_color_palette(n_color)).astype('int')
  f = open(filename, 'w')
  f.write('COFF\n{} {} 0\n'.format(vertices.shape[0], faces.shape[0]))
  for i in range(vertices.shape[0]):
    f.write('{} {} {}\n'.format(vertices[i][0],vertices[i][1],vertices[i][2]))
  for i in range(faces.shape[0]):
    f.write('3 {} {} {} '.format(faces[i][0], faces[i][1], faces[i][2]))
    if f_mask[i] == -1:
      f.write('255 255 255\n')
    else:
      f.write('{} {} {}\n'.format(colormap[f_mask[i]][0], colormap[f_mask[i]][1], colormap[f_mask[i]][2]))
    # f.write('3 {} {} {}\n'.format(faces[i][0], faces[i][1], faces[i][2]))

def export_patches_off(patch_points, export_off_filename):
  patch_points_reshape = np.reshape(patch_points, [-1, 3])
  # assert(patch_points_reshape.shape[0] % 100 == 0)
  num_of_valid_patches = patch_points_reshape.shape[0] // (points_per_patch_dim * points_per_patch_dim)
  # with open(export_obj_filename, "w") as wf:
  #   wf.write("mtllib default.mtl\n")
  #   num_mtl_in_lib = 15
  
  dim_square = (points_per_patch_dim * points_per_patch_dim)
  verts = []
  faces = []
  face_mask = []
  for patch_idx in range(num_of_valid_patches):
    # wf.write("usemtl m{}\n".format(patch_idx % num_mtl_in_lib))
    for point in patch_points_reshape[dim_square*patch_idx:dim_square*patch_idx+dim_square]:
      # wf.write("v %lf %lf %lf\n" %(point[0], point[1], point[2]))
      verts.append([point[0], point[1], point[2]])
    # face_displacement = patch_idx*100 + 1
    face_displacement = patch_idx*dim_square
    for row_idx in range(points_per_patch_dim - 1):
      for col_idx in range(points_per_patch_dim):
        vert_idx = face_displacement + row_idx*points_per_patch_dim + col_idx
        if(col_idx != 0):
          # wf.write("f %d %d %d\n" %(vert_idx, vert_idx-1, vert_idx+10))
          faces.append([vert_idx, vert_idx-1, vert_idx+points_per_patch_dim])
          face_mask.append(patch_idx)
        if(col_idx != points_per_patch_dim - 1):
          # wf.write("f %d %d %d\n" %(vert_idx, vert_idx+10, vert_idx+11))  
          faces.append([vert_idx, vert_idx+points_per_patch_dim, vert_idx+points_per_patch_dim + 1])
          face_mask.append(patch_idx)
  
  verts = np.array(verts)
  faces = np.array(faces)
  face_mask = np.array(face_mask)
  save_mesh_off(export_off_filename, verts, faces, face_mask)


def eval_pipeline(flag_eval = True):  
  # fix the seed for reproducibility
  '''
  seed = args.seed + rank
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  '''
  
  #not considering rotation augmentation
  # train_data = train_data_loader(args.batch_size, voxel_dim=voxel_dim, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals, data_folder="/mnt/data/shilin/detr/ABC/test", with_distribute_sampler=False)
  # train_data = train_data_loader(args.batch_size, voxel_dim=voxel_dim, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals, data_folder="test_data_tmp", with_distribute_sampler=False)
  if args.quicktest:
    # train_data = train_data_loader(args.batch_size, voxel_dim=voxel_dim, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals,  data_folder="train_small",  with_distribute_sampler=False, flag_quick_test=args.quicktest, flag_noise=args.noise)#data_folder="/mnt/data/shilin/detr/ABC/train",
    train_data = train_data_loader(args.batch_size, voxel_dim=voxel_dim, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals,  data_folder="train_small",  with_distribute_sampler=False, flag_quick_test=args.quicktest, flag_noise=args.noise, flag_grid = args.patch_grid, flag_patch_uv=args.patch_uv, dim_grid = points_per_patch_dim, eval_res_cov = args.eval_res_cov)#data_folder="/mnt/data/shilin/detr/ABC/train",
  else:
    # train_data = train_data_loader(args.batch_size, voxel_dim=voxel_dim, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals,  data_folder="val_tmp", with_distribute_sampler=False, flag_quick_test=args.quicktest)#data_folder="/mnt/data/shilin/detr/ABC/test",
    # train_data = train_data_loader(args.batch_size, voxel_dim=voxel_dim, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals,  data_folder="/mnt/sdf1/haog/data/train_new", with_distribute_sampler=False, flag_quick_test=args.quicktest)#data_folder="/mnt/data/shilin/detr/ABC/test",
    vis_train_folder = "vis_train"
    vis_test_folder = "vis_test"
    test_folder = "/mnt/sdf1/haog/data/test_new"
    train_folder = "/mnt/sdf1/haog/data/train_new"
    if args.parsenet:
      # vis_train_folder = "vis_train_parsenet"
      # vis_train_folder = "vis_tmp"
      # vis_test_folder = "vis_test_parsenet"

      #train
      # vis_train_folder = "vis_train_perpatch"
      # vis_test_folder = "vis_test_perpatch"
      # vis_test_folder = "vis_test_tmp"


      vis_train_folder = "vis_train_perpatch_1120"
      vis_test_folder = "vis_test_perpatch_1120"
      # vis_test_folder = "vis_tmp"

      #partial
      # vis_test_folder = "partial_data_vis_test"
      # vis_test_folder = "vis_test_tmp"
      # vis_test_folder = "vis_test_perpatch_tmp"
      # test_folder = "/mnt/sdf1/haog/data/data_parsenet_perpatch_test"

      # test_folder = "/mnt/sdf1/haog/data/data_parsenet_perpatch_test_1120"

      # test_folder = "/mnt/sdf1/haog/data/tmp"

      # test_folder = "/mnt/sdf1/haog/data/data_parsenet_perpatch_train_partial"

      # test_folder = "/mnt/sdf1/haog/data/val_perpatch_900_partial"
      # test_folder = "/mnt/sdf1/haog/data/val_perpatch_900_1120"
      base_folder = "/mnt/sdf1/haog/data"
      if args.wsa:
        base_folder = "/D_data/haog/complexgen/data/test_data"
        # base_folder = "/D_data/haog/complexgen/data/test_data"

      # test_folder = "/mnt/sdf1/haog/data/data_parsenet_perpatch_test_1120"

      #ori version
      test_folder = os.path.join(base_folder, "data_parsenet_perpatch_test_1120")

      # test_folder = os.path.join(base_folder, "fusion360")
      # test_folder = os.path.join(base_folder, "globfit")
      # test_folder = os.path.join(base_folder, "1558")
      # test_folder = os.path.join(base_folder, "cube_rot")
      # test_folder = os.path.join(base_folder, "cube")
      # test_folder = os.path.join(base_folder, "realscan")
      # test_folder = os.path.join(base_folder, "union")
      # test_folder = os.path.join(base_folder, "aimshape_poisson")





      # test_folder = os.path.join(base_folder, "fandisk")



      # test_folder = os.path.join(base_folder, "tmp0127")


      if args.evalrest:
        # test_folder = "/mnt/sdf1/haog/data/val_perpatch_900_rest_1120"
        test_folder = os.path.join(base_folder, "val_perpatch_900_rest_1120")
        # test_folder = "/mnt/sdf1/haog/data/tmp"
      # test_folder = "/mnt/sdf1/haog/data/tmp1223"
      # test_folder = "/mnt/sdf1/haog/data/tmp1224"
      # test_folder = "/mnt/sdf1/haog/data/val_perpatch_900_rest_1120"
      # test_folder = "tmp_small"
      # test_folder = "/mnt/sdf1/haog/data/data_parsenet_perpatch_train_1120"
      # test_folder = "/mnt/sdf1/haog/data/val_perpatch_900"

      #part version
      if args.part >= 0:
        # test_folder = "/mnt/sdf1/haog/data/data_parsenet_perpatch_test_1120_part{}".format(args.part)
        test_folder = os.path.join(base_folder, "data_parsenet_perpatch_test_1120_part{}".format(args.part))

      if args.partial:
        # test_folder = "/mnt/sdf1/haog/data/data_parsenet_perpatch_test_partial/partial"
        test_folder = os.path.join(base_folder, "data_parsenet_perpatch_test_partial/partial")

      # test_folder = os.path.join(base_folder, "aimshape")

      # test_folder = os.path.join(base_folder, "aimshape_poisson")
      test_folder = os.path.join(base_folder, "shapenet_camera1")
      # test_folder = os.path.join(base_folder, "shapenet_other")
      # test_folder = os.path.join(base_folder, "union_partial_poisson")



      #old version
      # vis_train_folder = "vis_train_parsenet"
      # vis_test_folder = "vis_test_parsenet"
      # test_folder = "/mnt/sdf1/haog/data/data_parsenet_fix_test"

      # test_folder = "/mnt/sdf1/haog/data/val_fix_900"

      # test_folder = "/mnt/sdf1/haog/data/data_parsenet_perpatch_test"
      # train_folder = "/mnt/sdf1/haog/data/data_parsenet_fix_train_1120"

      
    
    if args.vis_train:
      train_data = train_data_loader(args.batch_size, voxel_dim=voxel_dim, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals,  data_folder=vis_train_folder, with_distribute_sampler=False, flag_quick_test=args.quicktest, flag_noise=args.noise, flag_grid = args.patch_grid, flag_patch_uv=args.patch_uv, dim_grid = points_per_patch_dim, eval_res_cov = args.eval_res_cov)#data_folder="/mnt/data/shilin/detr/ABC/test",
    elif args.vis_test:
      train_data = train_data_loader(args.batch_size, voxel_dim=voxel_dim, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals,  data_folder=vis_test_folder, with_distribute_sampler=False, flag_quick_test=args.quicktest, flag_noise=args.noise, flag_grid = args.patch_grid, flag_patch_uv=args.patch_uv, dim_grid = points_per_patch_dim, eval_res_cov = args.eval_res_cov)#data_folder="/mnt/data/shilin/detr/ABC/test",
    elif args.eval_train:
      train_data = train_data_loader(args.batch_size, voxel_dim=voxel_dim, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals,  data_folder=train_folder, with_distribute_sampler=False, flag_quick_test=args.quicktest, flag_noise=args.noise, flag_grid = args.patch_grid, flag_patch_uv=args.patch_uv, dim_grid = points_per_patch_dim, eval_res_cov = args.eval_res_cov)#data_folder="/mnt/data/shilin/detr/ABC/test", 
    else:
      train_data = train_data_loader(args.batch_size, voxel_dim=voxel_dim, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals,  data_folder=test_folder, with_distribute_sampler=False, flag_quick_test=args.quicktest, flag_noise=args.noise, flag_grid = args.patch_grid, flag_patch_uv=args.patch_uv, dim_grid = points_per_patch_dim, eval_res_cov = args.eval_res_cov)#data_folder="/mnt/data/shilin/detr/ABC/test",
      # train_data = train_data_loader(args.batch_size, voxel_dim=voxel_dim, feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals,  data_folder="/mnt/sdf1/haog/data/train_small", with_distribute_sampler=False, flag_quick_test=args.quicktest, flag_noise=args.noise)#data_folder="/mnt/data/shilin/detr/ABC/test",
      

  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  disable_aux_loss_output = True
  #torch.autograd.set_detect_anomaly(True)
  tf.compat.v1.disable_eager_execution()
  
  if args.no_tri:
    model_shape, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion = build_unified_model(device, flag_eval)
  else:
    model_shape, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion = build_unified_model_tripath(device, flag_eval)

  model_without_ddp = model_shape
  
  param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]}]
  
  corner_topo_params = {n:p for n, p in model_without_ddp.named_parameters() if p.requires_grad and 'corner_model.corner_topo_embed' in n}
  corner_geometry_params = {n:p for n, p in model_without_ddp.named_parameters() if p.requires_grad and 'corner_model.corner_position_embed' in n}
  
  optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
  
  log_dir, obj_dir, checkpoint_dir = prepare_experiment_folders(args.experiment_name)
  if(os.path.exists("experiments/default.mtl") and not os.path.exists(os.path.join(obj_dir, "default.mtl"))):
    os.system("cp experiments/default.mtl {}".format(obj_dir))

  experiment_dir = os.path.join("experiments", args.experiment_name)

  test_folder = 'test_obj'
  if args.evalrest:
    test_folder = 'test_obj_rest'
  if args.vis_train:
    test_folder = 'vis_train'
  elif args.vis_test:
    test_folder = 'vis_test'
  elif args.eval_train:
    test_folder = 'train_obj'

  if args.vis_inter_layer != -1:
    test_folder = test_folder + '_vislayer{}'.format(args.vis_inter_layer)

  testobj_dir = os.path.join(experiment_dir, test_folder)
  if(not os.path.exists(testobj_dir)): os.mkdir(testobj_dir)
  if(os.path.exists("experiments/default.mtl") and not os.path.exists(os.path.join(testobj_dir, "default.mtl"))):
    os.system("cp experiments/default.mtl {}".format(testobj_dir))
  
  start_iterations = 0
  print("Try to restore from checkpoint")
  if(args.checkpoint_path is not None):
    if(os.path.exists(args.checkpoint_path)):
      print("resume training using {}".format(args.checkpoint_path))
      checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
      model_without_ddp.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
      start_iterations = checkpoint['epoch'] + 1
    else:
      print("specified checkpoint file cannot be found: {}".format(args.checkpoint_path))
  elif(args.enable_automatic_restore):
    print("trying to restore automatically")
    all_ckpt = os.listdir(checkpoint_dir)
    restore_ckpt = None
    restore_ckpt_epoches = -1
    for ckpt_file in all_ckpt:
      if(ckpt_file.endswith(".pth")):
        ckpt_epoches = int(ckpt_file.split('_')[1].split(".")[0])
        if(ckpt_epoches > restore_ckpt_epoches):
          restore_ckpt_epoches = ckpt_epoches
          restore_ckpt = os.path.join(checkpoint_dir, ckpt_file)
    if(restore_ckpt is not None):
      print("find available ckpt file:", restore_ckpt)
      checkpoint = torch.load(restore_ckpt, map_location='cpu')
      model_without_ddp.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
      start_iterations = checkpoint['epoch'] + 1
    else:
      print("cannot find available ckpt file")
  
  if not args.evalfinal and not args.evaltopo:
    return model_evaluation(model_without_ddp, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion, train_data, device, start_iterations, flag_output = not args.no_output, test_folder = test_folder)
  elif args.evalfinal:
    # return model_evaluation_from_file(model_without_ddp, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion, train_data, device, start_iterations, flag_output = not args.no_output)

    return model_evaluation_yaml(model_without_ddp, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion, train_data, device, start_iterations, flag_output = not args.no_output)

  elif args.evaltopo:
    return model_evaluation_topo(model_without_ddp, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion, train_data, device, start_iterations, flag_output = not args.no_output)

def eval_files():
  #eval corner curve patch distance from predicted files
  import trimesh
  from numpy import linalg as LA
  from scipy.spatial import cKDTree
  # def numpy_chamfer_dist(src, tgt):
  #   #both src and tgt should be in shape (n,3)
  #   len_src = src.shape[0]
  #   len_tgt = tgt.shape[0]
  #   # src_expand = np.repeat(src, len_tgt).reshape(len_src, len_tgt, 3)
  #   # print("len src: ", len_src)
  #   src_expand = np.tile(src.reshape(-1,1,3), (1, len_tgt, 1))
  #   tgt_expand = np.tile(tgt.reshape(1,-1,3), (len_src, 1,1))
  #   assert(src_expand.shape == tgt_expand.shape)
  #   dist_mat = LA.norm(src_expand - tgt_expand, axis = -1)
  #   dist_forward = dist_mat.min(0).mean()
  #   dist_backward = dist_mat.min(-1).mean()
  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  matcher_corner = build_matcher_corner(args, flag_eval = True)
  matcher_curve = build_matcher_curve(args, flag_eval = True)
  matcher_patch = build_matcher_patch(args, flag_eval = True)
  #criterion
  
  corner_losses = ['cd']
  curve_losses = ['cd']
  patch_losses = ['cd']

  corner_loss_criterion = SetCriterion_Corner(matcher_corner, {}, corner_eos_coef_cal, corner_losses).to(device)
  curve_loss_criterion = SetCriterion_Curve(matcher_curve, {}, curve_eos_coef_cal, curve_losses).to(device)
  patch_loss_criterion = SetCriterion_Patch(matcher_patch, {}, patch_eos_coef_cal, patch_losses).to(device)

  def numpy_chamfer_dist(src, tgt):
    kdtree1 = cKDTree(src)
    dist_tgt, idx_tgt = kdtree1.query(tgt)
    kdtree2 = cKDTree(tgt)
    dist_src, idx_src = kdtree2.query(src)
    return (dist_src.mean() + dist_tgt.mean()) / 2.0
  gt_data_path = "/mnt/sdf1/haog/data/test_new"

  path = os.path.join("experiments", args.experiment_name, "test_obj")
  allfs = os.listdir(path)
  all_valid_id = []
  for f in allfs:
    if f.endswith("_opt_post_mix_step5.obj"):
      all_valid_id.append(int(f.split("_")[0]))
  
  print('valid size: ', len(all_valid_id))
  
  allstats = []
  sum_dict = {}
  
  sample_id = []
  sum_dict["corner_cd"] = 0.0
  sum_dict["curve_cd"] = 0.0
  sum_dict["patch_cd"] = 0.0

  # samplecount = 0
  nopatchcount = 0
  for i in range(len(all_valid_id)):
  # for i in range(3):
    cur_id = all_valid_id[i]
    if cur_id == 14964:
      continue
    print("id: ", cur_id)
    gt_file = os.path.join(gt_data_path, '{:08d}.pkl'.format(cur_id))
    #get corner curve and patch
    f = open(gt_file, 'rb')
    data = pickle.load(f)
    curve_list = []
    corner_id = []
    corner_list = []
    for curveid, curve in enumerate(data['curves']):
      # print('pts shape: ', curve['points'][::3].shape)
      curve_list.append(curve['points'][::3])
      if not curve['start_vert_idx'] in corner_id:
        corner_id.append(curve['start_vert_idx'])
        corner_list.append(curve['points'][0])
      if not curve['end_vert_idx'] in corner_id:
        corner_id.append(curve['end_vert_idx'])
        corner_list.append(curve['points'][-1])
    curve = np.concatenate(curve_list)
    corner = np.concatenate(corner_list).reshape(-1,3)

    allpatch = data['surface_points'] #(10000, 6)
    # print('patch shape: ', allpatch.shape)
    # np.savetxt('gtpatch.xyz', allpatch)
    # np.savetxt('gtcurve.xyz', curve)
    # np.savetxt('gtcorner.xyz', corner)
    #predinfo
    pred_curve_file = os.path.join(path, "{}_opt_post_mix_curves_5.ply".format(cur_id))
    pred_corner_file = os.path.join(path, "{}_opt_post_mix_corners_5.ply".format(cur_id))
    pred_patch_file = os.path.join(path, "{}_opt_post_mix_step5.obj".format(cur_id))
    pred_patch_mask = os.path.join(path, "{}_opt_post_mix_facemask.txt".format(cur_id))
    pred_corres = os.path.join(path, "{}_opt_post_mix_final_topo.txt".format(cur_id))

    if args.fittingonce:
      pred_curve_file = os.path.join(path, "{}_opt_post_mix_curves_init.ply".format(cur_id))
      pred_corner_file = os.path.join(path, "{}_opt_post_mix_corners_init.ply".format(cur_id))
    
    pred_curve = load_ply_pts(pred_curve_file)
    pred_corner = load_ply_pts(pred_corner_file)
    # np.savetxt('predcurve.xyz', pred_curve)
    # np.savetxt('predcorner.xyz', pred_corner)

    # a = np.zeros(21).reshape(7,3)
    # b = a + 1
    # dist = numpy_chamfer_dist(a,b)
    # print("dist: ", dist)
    dist_curve = numpy_chamfer_dist(pred_curve, curve)
    dist_corner = 0.0
    if pred_corner.shape[0] > 0 and corner.shape[0] > 0:
      dist_corner = numpy_chamfer_dist(pred_corner, corner)

    dist_patch = 0.0
    n_sample = 10000
    # print('mesh file: ', pred_patch_file)
    mesh = trimesh.load(pred_patch_file)
    #if there are no faces, set distance to 0
    meshtype = str(type(mesh))
    if meshtype.split(' ')[-1].split('.')[-1][:-2] != 'PointCloud':
      pc, idx = mesh.sample(n_sample, return_index=True)
      # print('pc shape: ', pc.shape)
      dist_patch = numpy_chamfer_dist(pc, allpatch[:, :3])
      # np.savetxt(os.path.join(path, "{}_opt_post_mix_step5.xyz".format(cur_id)), pc)
      # np.savetxt(os.path.join(path, "{}_opt_post_mix_gt.xyz".format(cur_id)), allpatch[:, :3])
      # np.savetxt('gt_patch.xyz', allpatch[:, :3])
    else:
      nopatchcount += 1
    
    #get patch point cloud
    # pred_data = {}
    

    sum_dict["corner_cd"] += dist_corner
    sum_dict["curve_cd"] += dist_curve
    sum_dict["patch_cd"] += dist_patch
    
    #to be computed: corner/curve/patch precision/recall topo error

    allstats.append([dist_corner, dist_curve, dist_patch])
    sample_id.append(cur_id)

    # print("corner dist: ", dist_corner)
    # print("dist curve: ", dist_curve)
    
    # curvet = torch.tensor(curve,dtype=torch.float32).view(1,-1,3)
    # cornert = torch.tensor(corner,dtype=torch.float32).view(1,-1,3)
    # pred_curvet = torch.tensor(pred_curve,dtype=torch.float32).view(1,-1,3)
    # pred_cornert = torch.tensor(pred_corner,dtype=torch.float32).view(1,-1,3)

    # from chamferdist import ChamferDistance #distance not corrected
    # chamferDist = ChamferDistance()
    # # dist_curve = chamferDist(curvet, pred_curvet)
    # dist_curve = chamferDist(curvet, curvet + 1.0)
    # dist_corner = chamferDist(cornert, pred_cornert)
    # print('cdist: {} {}'.format(dist_curve, dist_corner))

  for k in sum_dict:
      sum_dict[k] = sum_dict[k] / len(sample_id)

  sample_id.append("mean")
  allstats.append(list(sum_dict.values()))
  print('samples with no patch: ', nopatchcount)
  import pandas as pd
  ## convert your array into a dataframe
  df = pd.DataFrame(np.array(allstats))
  df.columns = list(sum_dict.keys())
  df.index = sample_id
  #title_row = pd.Series()
  #row_df = pd.DataFrame([title_row])
  #df = pd.concat([row_df, df], ignore_index=True)
  
  ## save to xlsx file
  filepath = os.path.join(path, 'final_triplecd.xlsx')
  df.to_excel(filepath, index=True)  


def pipeline_abc(rank, world_size):
  torch.autograd.set_detect_anomaly(True)
  dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23257', world_size=world_size, rank=rank)

  assert(is_dist_avail_and_initialized())
  assert(get_world_size() == world_size)
  
  # fix the seed for reproducibility
  '''
  seed = args.seed + rank
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  '''

  if True:
    if args.quicktest:
      train_data, distribute_sampler = train_data_loader(args.batch_size, voxel_dim=voxel_dim, data_folder="data/train_small", feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, random_angle = args.random_angle, with_normal=args.input_normal_signals, flag_quick_test=args.quicktest,flag_noise=args.noise, flag_grid = args.patch_grid, num_angle = args.num_angles, flag_patch_uv=args.patch_uv, dim_grid = points_per_patch_dim, eval_res_cov = args.extra_single_chamfer) #


      #ori version
      val_data, val_data_sampler = train_data_loader(args.batch_size, voxel_dim=voxel_dim, data_folder="data/train_small", feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals, flag_quick_test=False, flag_noise=args.noise, flag_grid = args.patch_grid, num_angle = args.num_angles, flag_patch_uv=args.patch_uv, dim_grid = points_per_patch_dim, eval_res_cov = args.extra_single_chamfer)#data_folder="/mnt/data/shilin/detr/ABC/train",

      #modified version on 1012
      # val_data, val_data_sampler = train_data_loader(args.batch_size, voxel_dim=voxel_dim, data_folder="/mnt/sdf1/haog/data/val_fix_900", feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals, flag_quick_test=False,flag_noise=args.noise, flag_grid = args.patch_grid, num_angle = args.num_angles, flag_patch_uv=args.patch_uv)#data_folder="/mnt/data/shilin/detr/ABC/train",

    else:
      #distrubted sampler only used for set epoches
      if not args.data_medium:
        # train_data, distribute_sampler = train_data_loader(args.batch_size, voxel_dim=voxel_dim, data_folder="/mnt/sdf1/haog/data/train_new", feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals, flag_quick_test=args.quicktest,flag_noise=args.noise) #"/mnt/data/shilin/detr/ABC/train"
        # train_data, distribute_sampler = train_data_loader(args.batch_size, voxel_dim=voxel_dim, data_folder="/mnt/sdf1/haog/data/train_new_aug", feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals, flag_quick_test=args.quicktest) #"/mnt/data/shilin/detr/ABC/train"
        # train_data, distribute_sampler = train_data_loader(args.batch_size, voxel_dim=voxel_dim, data_folder="/mnt/sdf1/haog/data/train_new_50k", feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment,random_angle = args.random_angle, with_normal=args.input_normal_signals, flag_quick_test=args.quicktest) #"/mnt/data/shilin/detr/ABC/train"
        # train_data, distribute_sampler = train_data_loader(args.batch_size, voxel_dim=voxel_dim, data_folder="/mnt/sdf1/haog/data/data_parsenet_train_oripkl", feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment,random_angle = args.random_angle, with_normal=args.input_normal_signals, flag_quick_test=args.quicktest) #"/mnt/data/shilin/detr/ABC/train"
        # train_data, distribute_sampler = train_data_loader(args.batch_size, voxel_dim=voxel_dim, data_folder="/mnt/sdf1/haog/data/data_parsenet_fix_train", feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment,random_angle = args.random_angle, with_normal=args.input_normal_signals, flag_quick_test=args.quicktest,flag_grid = args.patch_grid, num_angle = args.num_angles) #"/mnt/data/shilin/detr/ABC/train"
        # train_data, distribute_sampler = train_data_loader(args.batch_size, voxel_dim=voxel_dim, data_folder="/mnt/sdf1/haog/data/tmp", feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment,random_angle = args.random_angle, with_normal=args.input_normal_signals, flag_quick_test=args.quicktest,flag_grid = args.patch_grid, num_angle = args.num_angles) #"/mnt/data/shilin/detr/ABC/train"
        # train_data, distribute_sampler = train_data_loader(args.batch_size, voxel_dim=voxel_dim, data_folder="/mnt/sdf1/haog/data/data_parsenet_perpatch_train", feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment,random_angle = args.random_angle, with_normal=args.input_normal_signals, flag_quick_test=args.quicktest,flag_grid = args.patch_grid, num_angle = args.num_angles, flag_patch_uv=args.patch_uv, dim_grid = points_per_patch_dim, eval_res_cov = args.extra_single_chamfer) #"/mnt/data/shilin/detr/ABC/train"

        train_data, distribute_sampler = train_data_loader(args.batch_size, voxel_dim=voxel_dim, data_folder="/mnt/sdf1/haog/data/data_parsenet_perpatch_train_partial", feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment,random_angle = args.random_angle, with_normal=args.input_normal_signals, flag_quick_test=args.quicktest,flag_grid = args.patch_grid, num_angle = args.num_angles, flag_patch_uv=args.patch_uv, dim_grid = points_per_patch_dim, eval_res_cov = args.extra_single_chamfer) #"/mnt/data/shilin/detr/ABC/train"

        

      else:
        train_data, distribute_sampler = train_data_loader(args.batch_size, voxel_dim=voxel_dim, data_folder="/mnt/sdf1/haog/data/train_medium", feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment,random_angle = args.random_angle, with_normal=args.input_normal_signals, flag_quick_test=args.quicktest,flag_noise=args.noise, flag_grid = args.patch_grid, num_angle = args.num_angles, flag_patch_uv=args.patch_uv, dim_grid = points_per_patch_dim, eval_res_cov = args.extra_single_chamfer) #"/mnt/data/shilin/detr/ABC/train"
      if not args.patch_grid:
        val_data, val_data_sampler = train_data_loader(args.batch_size, voxel_dim=voxel_dim, data_folder="/mnt/sdf1/haog/data/val_new", feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals, flag_quick_test=False,flag_noise=args.noise, flag_grid = args.patch_grid, num_angle = args.num_angles, flag_patch_uv=args.patch_uv, dim_grid = points_per_patch_dim, eval_res_cov = args.extra_single_chamfer)#data_folder="/mnt/data/shilin/detr/ABC/train",
      else:
        # val_data, val_data_sampler = train_data_loader(args.batch_size, voxel_dim=voxel_dim, data_folder="/mnt/sdf1/haog/data/val_fix_64", feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals, flag_quick_test=False,flag_noise=args.noise, flag_grid = args.patch_grid, num_angle = args.num_angles, flag_patch_uv=args.patch_uv)#data_folder="/mnt/data/shilin/detr/ABC/train",
        val_data, val_data_sampler = train_data_loader(args.batch_size, voxel_dim=voxel_dim, data_folder="/mnt/sdf1/haog/data/val_perpatch_900", feature_type=args.input_feature_type, pad1s=not args.backbone_feature_encode, rotation_augmentation=args.rotation_augment, with_normal=args.input_normal_signals, flag_quick_test=False,flag_noise=args.noise, flag_grid = args.patch_grid, num_angle = args.num_angles, flag_patch_uv=args.patch_uv, dim_grid = points_per_patch_dim, eval_res_cov = args.extra_single_chamfer)#data_folder="/mnt/data/shilin/detr/ABC/train",
  
  torch.cuda.set_device(rank)
  device = 'cuda:{}'.format(rank) if torch.cuda.is_available() else 'cpu'
  disable_aux_loss_output = True
  #torch.autograd.set_detect_anomaly(True)
  tf.compat.v1.disable_eager_execution()
  if args.no_tri:
    model_shape, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion = build_unified_model(device)
  else:
    model_shape, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion = build_unified_model_tripath(device)
  
  model_shape = torch.nn.parallel.DistributedDataParallel(model_shape, device_ids=[rank]) #,find_unused_parameters=True
  model_without_ddp = model_shape.module
  
  param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]}]
  
  if args.ori_topo:
    corner_topo_params = {n:p for n, p in model_without_ddp.named_parameters() if p.requires_grad and 'corner_model.corner_topo_embed' in n}
  else:
    corner_topo_params = {n:p for n, p in model_without_ddp.named_parameters() if p.requires_grad and ('corner_model.corner_topo_embed_curve' in n or 'curve_model.curve_topo_embed_corner' in n or 'corner_model.corner_topo_embed_patch' in n or 'patch_model.patch_topo_embed_corner' in n)}
  corner_geometry_params = {n:p for n, p in model_without_ddp.named_parameters() if p.requires_grad and 'corner_model.corner_position_embed' in n}
  #print(corner_topo_params)
  #print(corner_geometry_params)
  
  optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma = 0.5)
  
  if(rank == 0):
    log_dir, obj_dir, checkpoint_dir = prepare_experiment_folders(args.experiment_name)
    if(os.path.exists("experiments/default.mtl") and not os.path.exists(os.path.join(obj_dir, "default.mtl"))):
      os.system("cp experiments/default.mtl {}".format(obj_dir))
  
  dist.barrier()
  log_dir, obj_dir, checkpoint_dir = prepare_experiment_folders(args.experiment_name)
  exp_dir = os.path.join("experiments", args.experiment_name)
  
  start_iterations = 0
  if(rank == 0):
    print("Try to restore from checkpoint")
  

  if(args.checkpoint_path is not None):
    if(os.path.exists(args.checkpoint_path)):
      print("resume training using {}".format(args.checkpoint_path))
      checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
      model_without_ddp.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
      start_iterations = checkpoint['epoch'] + 1
    else:
      print("specified checkpoint file cannot be found: {}".format(args.checkpoint_path))
  elif(args.enable_automatic_restore):
    print("trying to restore automatically")
    all_ckpt = os.listdir(checkpoint_dir)
    restore_ckpt = None
    restore_ckpt_epoches = -1
    for ckpt_file in all_ckpt:
      if(ckpt_file.endswith(".pth")):
        ckpt_epoches = int(ckpt_file.split('_')[1].split(".")[0])
        if(ckpt_epoches > restore_ckpt_epoches):
          restore_ckpt_epoches = ckpt_epoches
          restore_ckpt = os.path.join(checkpoint_dir, ckpt_file)
    if(restore_ckpt is not None):
      print("find available ckpt file:", restore_ckpt)
      checkpoint = torch.load(restore_ckpt, map_location='cpu')
      model_without_ddp.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
      start_iterations = checkpoint['epoch'] + 1
    else:
      print("cannot find available ckpt file")
  
  if(args.eval):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return model_evaluation(model_without_ddp, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion, train_data, device, start_iterations)
    # return model_evaluation(model_without_ddp, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion)
  
  training_range = range(start_iterations, args.max_training_iterations)
  if(not running_onCluster and rank == 0): training_range = tqdm(training_range)
  
  #testing for validation part
  # val_data_sampler.set_epoch(0)
  # distribute_sampler.set_epoch(0)
  # summary_ref = {}
  # summary_ref['corner_curve_geom'] = 0.0
  # summary_ref['corner_curve_topo'] = 0.0
  # summary_ref['corner_loss_ce'] = 0.0
  # summary_ref['corner_loss_geometry'] = 0.0
  # summary_ref['corner_valid_accuracy'] = 0.0
  # summary_ref['curve_loss_curve_closed'] = 0.0
  # summary_ref['curve_loss_curve_type_ce'] = 0.0
  # summary_ref['curve_loss_geometry'] = 0.0
  # summary_ref['curve_loss_valid_ce'] = 0.0
  # summary_ref['curve_type_accuracy'] = 0.0
  # summary_ref['curve_valid_accuracy'] = 0.0
  # summary_ref['patch_curve_topo'] = 0.0
  # summary_ref['patch_loss_geometry'] = 0.0
  # summary_ref['patch_loss_patch_type_ce'] = 0.0
  # summary_ref['patch_loss_valid_ce'] = 0.0
  # summary_ref['patch_type_accuracy'] = 0.0
  # summary_ref['patch_valid_accuracy'] = 0.0
  # summary_ref['total_loss'] = 0.0


  # val_summary_loss_dict = get_val_summary_dict(model_shape, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion, val_data, device, 0, summary_ref)
  # val_summary_loss_dict_reduced = reduce_dict(val_summary_loss_dict)
  # return

  if(rank == 0):
    summary_writer = tf.compat.v1.summary.FileWriter(log_dir)
    print("Start Training")  
    print("train data size {}".format(len(train_data)))
  data_loader_iterator = iter(train_data)
  cur_epochs = 0
  val_epochs = 0
  distribute_sampler.set_epoch(cur_epochs)
  for train_iter in training_range:
      #if(train_iter % 500 == 0): torch.cuda.empty_cache()
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
    if True:
      model_shape.train()
      corner_loss_criterion.train()
      curve_loss_criterion.train()
      patch_loss_criterion.train()
      ############# Prepare Input #############
      t0 = time.time()
      try:
        data_item = next(data_loader_iterator)
      except StopIteration:
        data_loader_iterator = iter(train_data)
        data_item = next(data_loader_iterator)
        cur_epochs += 1
        distribute_sampler.set_epoch(cur_epochs)
      locations = data_item[0].to(device)
      features = data_item[1].to(device)
      corner_points = data_item[2].to(device)
      corner_batch_idx = data_item[3].to(device)
      #print("locations with shape: {}".format(locations.shape))
      #print("features with shape: {}".format(features.shape))
      #print("corner points with shape: {}".format(corner_points.shape))
      batch_sample_id = data_item[5]
      target_curves_list = data_item[6]
      target_patches_list = data_item[7]
      # print('train_iter: ', train_iter)
      # print('batch id: ', batch_sample_id)
      # print('patch key outside: ', target_patches_list[0].keys())
      # print('patch outside patch_points: ', target_patches_list[0]['patch_points']) #list of tensor
      #not using lr drop temporarily
      # if (train_iter + 1) % args.lr_drop == 0:
      #   lr_scheduler.step()
      # for bsi in batch_sample_id:
      #   print('sample id: ', bsi)
      #convert target_curves_list and target_patches_list to cuda tensors
      for i in range(len(target_curves_list)):
        for k in target_curves_list[i]:
          target_curves_list[i][k] = target_curves_list[i][k].to(device)
      
      for i in range(len(target_patches_list)):
        for k in target_patches_list[i]:
          if k == 'patch_points' or k == 'patch_normals' or k == 'patch_pcs':
            for j in range(len(target_patches_list[i][k])): #list
              target_patches_list[i][k][j] = target_patches_list[i][k][j].to(device)
          else:
            target_patches_list[i][k] = target_patches_list[i][k].to(device)

      #supervision
      batch_corner_numbers = []
      for i in range(args.batch_size):
        batch_corner_numbers.append((corner_batch_idx==i).sum())
      target_corner_points_list = torch.split(corner_points, batch_corner_numbers)
      t1 = time.time()
      #print("{}s elapsed for data preparation".format(t1-t0))
      if(train_iter < profile_iter and perform_profile and train_iter > 2):
        profile_dict['data_preparation'].append(t1 - t0)
      
      #forward
      t0 = time.time()
      sparse_locations, corner_predictions, curve_predictions, patch_predictions = model_shape(locations, features)
      t1 = time.time()

      #print("{}s elapsed for network forwarding".format(t1-t0))
      if(train_iter < profile_iter and perform_profile and train_iter > 2):
        profile_dict['network_forwarding'].append(t1 - t0)
      
      def export_curves(curve_points, points_number_per_curve, output_obj_file):
        curve_points = np.reshape(curve_points, [-1,3])
        with open(output_obj_file, "w") as wf:
          for point in curve_points:
            wf.write("v {} {} {}\n".format(point[0], point[1], point[2]))
          for i in range(curve_points.shape[0]):
            if(i % points_number_per_curve == (points_number_per_curve - 1)):
              continue
            wf.write("l {} {}\n".format(i+1, i+2))
      
      # if(rank == 0 and (train_iter % 500 == 0 or (train_iter % 500 == 0  and train_iter < 1000 and not running_onCluster))):
      if False:
        #not save intermediate results anymore
        # print(batch_sample_id[0])
        #visualization
        '''      
        if not running_onCluster:
          voxel_points = data_item[8][:,:-1]
          assert(voxel_points.shape[1] == 3 or voxel_points.shape[1] == 6)
          if(voxel_points.shape[1] == 3):
            np.savetxt(os.path.join(obj_dir, "input_voxel_points_{:06d}.xyz".format(train_iter)), voxel_points)
          else:
            write_ply(os.path.join(obj_dir, "input_voxel_points_{:06d}.ply".format(train_iter)), voxel_points)
        '''
        
        input_pointcloud = data_item[4][0]
        np.savetxt(os.path.join(obj_dir, "input_{:06d}_{:06d}.xyz".format(train_iter, int(batch_sample_id[0].replace(".pkl" ,"")))), input_pointcloud)
        
        #curves
        labels = torch.argmax(curve_predictions['pred_curve_logits'].softmax(-1)[0], dim=-1).cpu().numpy()#100, 0 or 1
        pred_curve_type = curve_type_list[torch.argmax(curve_predictions['pred_curve_type'].softmax(-1)[0], dim=-1).cpu().numpy().astype(np.int32)[np.where(labels == 0)]].tolist()
        gt_curve_type = curve_type_list[target_curves_list[0]['labels'].cpu().numpy().astype(np.int32)].tolist()
        with open(os.path.join(obj_dir, "pred_curves_type_{:06d}.txt".format(train_iter)), "w") as f:
          f.write("{}\n".format(batch_sample_id[0]))
          f.write("{} {}\n".format(len(pred_curve_type), len(gt_curve_type)))
          for item in pred_curve_type: f.write("{}\n".format(item))
          f.write("==============================\n")
          for item in gt_curve_type: f.write("{}\n".format(item))
        
        curve_points = curve_predictions['pred_curve_points'][0].detach().cpu().numpy()
        effective_curve_points = np.reshape(curve_points[np.where(labels == 0)], [-1,3]) #100 34 3
        #np.savetxt(os.path.join(obj_dir, "pred_curves_{:06d}.xyz".format(train_iter)), effective_curve_points)
        export_curves(effective_curve_points, points_per_curve, os.path.join(obj_dir, "pred_curves_{:06d}.obj".format(train_iter)))
        
        empty_curve_points = np.reshape(curve_points[np.where(labels == 1)], [-1,3])
        #np.savetxt(os.path.join(obj_dir, "pred_curves_empty_{:06d}.xyz".format(train_iter)), empty_curve_points)
        export_curves(empty_curve_points, points_per_curve, os.path.join(obj_dir, "pred_curves_empty_{:06d}.obj".format(train_iter)))
        
        target_curve_points = np.reshape(target_curves_list[0]['curve_points'].cpu().numpy(), [-1,3])
        np.savetxt(os.path.join(obj_dir, "gt_curves_{:06d}.xyz".format(train_iter)), target_curve_points)
        export_curves(target_curve_points, points_per_curve, os.path.join(obj_dir, "gt_curves_{:06d}.obj".format(train_iter)))
        
        #corners
        labels = torch.argmax(corner_predictions['pred_logits'].softmax(-1)[0], dim=-1).cpu().numpy()
        corner_position = corner_predictions['pred_corner_position'][0].detach().cpu().numpy()
        effective_corner_position = corner_position[np.where(labels == 0)]
        np.savetxt(os.path.join(obj_dir, "pred_corner_{:06d}.xyz".format(train_iter)), effective_corner_position)
        
        empty_corner_position = corner_position[np.where(labels == 1)]
        np.savetxt(os.path.join(obj_dir, "pred_corner_empty_{:06d}.xyz".format(train_iter)), empty_corner_position)
        
        target_corner_position = target_corner_points_list[0].cpu().numpy()
        np.savetxt(os.path.join(obj_dir, "gt_corner_{:06d}.xyz".format(train_iter)), target_corner_position)
        
        #locations_out = (np.reshape(sparse_locations[0].cpu().numpy(), [-1,4])[:,1:] + 0.5) / out_voxel_dim - 0.5
        #np.savetxt(os.path.join(obj_dir, "input_sparse_voxel_{:06d}.xyz".format(train_iter)), locations_out)
        
        #patches
        patch_labels = torch.argmax(patch_predictions['pred_patch_logits'].softmax(-1)[0], dim=-1).cpu().numpy()
        
        pred_patch_type = patch_type_list[torch.argmax(patch_predictions['pred_patch_type'].softmax(-1)[0], dim=-1).cpu().numpy().astype(np.int32)[np.where(patch_labels == 0)]].tolist()
        gt_patch_type = patch_type_list[target_patches_list[0]['labels'].cpu().numpy().astype(np.int32)].tolist()
        with open(os.path.join(obj_dir, "pred_patches_type_{:06d}.txt".format(train_iter)), "w") as f:
          f.write("{} {}\n".format(len(pred_patch_type), len(gt_patch_type)))
          for item in pred_patch_type: f.write("{}\n".format(item))
          f.write("==============================\n")
          for item in gt_patch_type: f.write("{}\n".format(item))
        
        patch_points = patch_predictions['pred_patch_points'][0].detach().cpu().numpy() #in shape [100, 100*100, 3]
        effective_patch_points = patch_points[np.where(patch_labels == 0)]
        export_patches(effective_patch_points, os.path.join(obj_dir, "pred_patches_{:06d}.obj".format(train_iter)))
      
      if(rank == 0 and train_iter % args.ckpt_interval == 0):
        #save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_{:06d}.pth".format(train_iter))
        save_on_master({'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': train_iter,
                        'args': args,
                    }, checkpoint_path)
      
      t0 = time.time()
      max_norm = args.clip_max_norm
      #losses
      #for debugging:
      curve_predictions['sample_names'] = batch_sample_id
      curve_loss_dict, curve_matching_indices = curve_loss_criterion(curve_predictions, target_curves_list)
      
      #for debugging
      # curve_loss_dict, curve_matching_indices = curve_loss_criterion(outputs = curve_predictions, target_curves_list = target_curves_list, sample_names = batch_sample_id)

      curve_weight_dict = curve_loss_criterion.weight_dict
      corner_loss_dict, corner_matching_indices = corner_loss_criterion(corner_predictions, target_corner_points_list)
      corner_weight_dict = corner_loss_criterion.weight_dict
      patch_loss_dict, patch_matching_indices = patch_loss_criterion(patch_predictions, target_patches_list)
      patch_weight_dict = patch_loss_criterion.weight_dict

      corner_losses = sum(corner_loss_dict[k] * corner_weight_dict[k] for k in corner_loss_dict.keys() if k in corner_weight_dict)
      curve_losses = sum(curve_loss_dict[k] * curve_weight_dict[k] for k in curve_loss_dict.keys() if k in curve_weight_dict)
      patch_losses = sum(patch_loss_dict[k] * patch_weight_dict[k] for k in patch_loss_dict.keys() if k in patch_weight_dict)
      losses = corner_losses + curve_losses + patch_losses
      
      
      if(rank == 0 and train_iter % 500 == 0 and not running_onCluster):
        filename = os.path.join(obj_dir, "prediction_{:06d}.pkl".format(train_iter))
        output_prediction(filename, corner_predictions, curve_predictions, patch_predictions, \
          target_corner_points_list, target_curves_list, target_patches_list, corner_matching_indices, curve_matching_indices, patch_matching_indices, sample_id=batch_sample_id[0])
      
      
      if(rank >= 0):
        #accuracy of elements
        summary_loss_dict = {}
        for k in corner_loss_dict.keys():
          if k in corner_weight_dict:
            if(disable_aux_loss_output and "_aux_" in k):
              continue
            summary_loss_dict["corner_" + k] = corner_weight_dict[k] * corner_loss_dict[k].detach()
        
        for k in curve_loss_dict.keys():
          if k in curve_weight_dict:
            if(disable_aux_loss_output and "_aux_" in k):
              continue
            summary_loss_dict["curve_" + k] = curve_weight_dict[k] * curve_loss_dict[k].detach()
        
        for k in patch_loss_dict.keys():
          if k in patch_weight_dict:
            if(disable_aux_loss_output and "_aux_" in k):
              continue
            summary_loss_dict["patch_" + k] = patch_weight_dict[k] * patch_loss_dict[k].detach()
        
        #accuracy of corner and curve predictions
        summary_loss_dict['corner_valid_accuracy'] = corner_loss_dict['corner_prediction_accuracy']
        summary_loss_dict['curve_valid_accuracy'] = curve_loss_dict['valid_class_accuracy']
        summary_loss_dict['curve_type_accuracy'] = curve_loss_dict['type_class_accuracy']
        summary_loss_dict['patch_valid_accuracy'] = patch_loss_dict['valid_class_accuracy']
        summary_loss_dict['patch_type_accuracy'] = patch_loss_dict['type_class_accuracy']

        #overall valid accuracy
        summary_loss_dict['corner_valid_accuracy_overall'] = corner_loss_dict['corner_prediction_accuracy_overall']
        summary_loss_dict['curve_valid_accuracy_overall'] = curve_loss_dict['valid_class_accuracy_overall']
        summary_loss_dict['patch_valid_accuracy_overall'] = patch_loss_dict['valid_class_accuracy_overall']

      if not args.no_topo:
        if(args.curve_corner_geom_loss_coef > 0 or args.curve_corner_topo_loss_coef > 0):
          if not args.topo_acc:
            if args.ori_topo:
              curve_corner_matching_loss_geom, curve_corner_matching_loss_topo, all_zero_corners = \
                Curve_Corner_Matching_v2(corner_predictions, curve_predictions, target_corner_points_list, target_curves_list, corner_matching_indices['indices'], curve_matching_indices['indices'])
            else:
              curve_corner_matching_loss_geom, curve_corner_matching_loss_topo, all_zero_corners = \
                Curve_Corner_Matching_tripath(corner_predictions, curve_predictions, target_corner_points_list, target_curves_list, corner_matching_indices['indices'], curve_matching_indices['indices'])   
          else:
            #not consider ori_topo
            curve_corner_matching_loss_geom, curve_corner_matching_loss_topo, all_zero_corners, curve_corner_topo_acc = \
              Curve_Corner_Matching_tripath(corner_predictions, curve_predictions, target_corner_points_list, target_curves_list, corner_matching_indices['indices'], curve_matching_indices['indices']) 
          losses += args.curve_corner_geom_loss_coef*curve_corner_matching_loss_geom + args.curve_corner_topo_loss_coef*curve_corner_matching_loss_topo
          if(rank >= 0):
            if(train_iter == 0): print("with curve corner correspondence loss")
            summary_loss_dict['corner_curve_topo'] = args.curve_corner_topo_loss_coef*curve_corner_matching_loss_topo.detach()
            summary_loss_dict['corner_curve_geom'] = args.curve_corner_geom_loss_coef*curve_corner_matching_loss_geom.detach()
            if args.topo_acc:
              summary_loss_dict['corner_curve_topoacc'] = curve_corner_topo_acc.detach()
          if('aux_outputs' in corner_predictions):
            assert('aux_outputs' in curve_predictions)
            for aux_id in range(len(corner_matching_indices['aux_outputs'])):
              curve_corner_matching_loss_geom_aux, curve_corner_matching_loss_topo_aux, _ = \
                Curve_Corner_Matching_v2(corner_predictions['aux_outputs'][aux_id], curve_predictions['aux_outputs'][aux_id], target_corner_points_list, target_curves_list, corner_matching_indices['aux_outputs'][aux_id]['indices'], curve_matching_indices['aux_outputs'][aux_id]['indices'])
              losses += args.curve_corner_geom_loss_coef*curve_corner_matching_loss_geom_aux + args.curve_corner_topo_loss_coef*curve_corner_matching_loss_topo_aux
              #summary_loss_dict['corner_curve_topo_aux_'+str(aux_id)] = args.curve_corner_topo_loss_coef*curve_corner_matching_loss_topo_aux.cpu().detach().numpy()
        elif(train_iter == 0 and rank == 0):
          print("without curve corner correspondence loss")
        
        # print("all zero corners: ", all_zero_corners)
        if(args.patch_curve_topo_loss_coef > 0):
          if not args.topo_acc:
            if args.ori_topo:
              patch_curve_matching_loss_topo = \
                Patch_Curve_Matching(curve_predictions, patch_predictions, target_curves_list, target_patches_list, curve_matching_indices['indices'], patch_matching_indices['indices'])
            else:
              patch_curve_matching_loss_topo = \
                Patch_Curve_Matching_tripath(curve_predictions, patch_predictions, target_curves_list, target_patches_list, curve_matching_indices['indices'], patch_matching_indices['indices'])
          else:
            patch_curve_matching_loss_topo, patch_curve_topo_acc = \
              Patch_Curve_Matching_tripath(curve_predictions, patch_predictions, target_curves_list, target_patches_list, curve_matching_indices['indices'], patch_matching_indices['indices'])
          losses += args.patch_curve_topo_loss_coef*patch_curve_matching_loss_topo
          if(rank >= 0):
            if(train_iter == 0): print("with patch curve correspondence loss")
            summary_loss_dict['patch_curve_topo'] = args.patch_curve_topo_loss_coef*patch_curve_matching_loss_topo.detach()
            if args.topo_acc:
              summary_loss_dict['patch_curve_topoacc'] = patch_curve_topo_acc.detach()
          if('aux_outputs' in corner_predictions):
            assert('aux_outputs' in curve_predictions)
            for aux_id in range(len(corner_matching_indices['aux_outputs'])):
              patch_curve_matching_loss_topo_aux = \
                Patch_Curve_Matching(curve_predictions['aux_outputs'][aux_id], patch_predictions['aux_outputs'][aux_id], target_curves_list, target_patches_list, curve_matching_indices['aux_outputs'][aux_id]['indices'], patch_matching_indices['aux_outputs'][aux_id]['indices'])
              losses += args.patch_curve_topo_loss_coef*patch_curve_matching_loss_topo_aux
        elif(train_iter == 0 and rank == 0):
          print("without patch curve correspondence loss")

        if (args.patch_corner_topo_loss_coef > 0):
          if not args.topo_acc:
            patch_corner_matching_loss_topo, curve_point_loss, curve_patch_loss, patch_close_loss = \
              Patch_Corner_Matching_tripath(corner_predictions, curve_predictions, patch_predictions, target_corner_points_list, target_curves_list, target_patches_list, corner_matching_indices['indices'],curve_matching_indices['indices'], patch_matching_indices['indices'])
          else:
            patch_corner_matching_loss_topo, curve_point_loss, curve_patch_loss, patch_close_loss, patch_corner_topo_acc = \
              Patch_Corner_Matching_tripath(corner_predictions, curve_predictions, patch_predictions, target_corner_points_list, target_curves_list, target_patches_list, corner_matching_indices['indices'],curve_matching_indices['indices'], patch_matching_indices['indices'])
          losses += args.patch_corner_topo_loss_coef*patch_corner_matching_loss_topo
          #four other term: 
          losses += args.topo_loss_coef * curve_point_loss
          losses += args.topo_loss_coef * curve_patch_loss
          losses += args.topo_loss_coef * patch_close_loss

          if(rank >= 0):
            if(train_iter == 0): print("with patch corner correspondence loss")
            summary_loss_dict['patch_corner_topo'] = args.patch_corner_topo_loss_coef*patch_corner_matching_loss_topo.detach()
            if args.topo_acc:
              summary_loss_dict['patch_corner_topoacc'] = patch_corner_topo_acc.detach()

            if not args.no_show_topo:
              summary_loss_dict['curve_point_loss'] = args.topo_loss_coef * curve_point_loss.detach()
              summary_loss_dict['curve_patch_loss'] = args.topo_loss_coef * curve_patch_loss.detach()
              summary_loss_dict['patch_close_loss'] = args.topo_loss_coef * patch_close_loss.detach()

      summary_loss_dict['total_loss'] = losses.detach()
      # summary_loss_dict_reduced = reduce_dict(summary_loss_dict)
      if(running_onCluster and train_iter % 200 == 0):
        summary_loss_dict_reduced = reduce_dict(summary_loss_dict)
      elif not running_onCluster:
        summary_loss_dict_reduced = reduce_dict(summary_loss_dict)
      # print('summary dict: ', summary_loss_dict)
      # print('summary dict reduced: ', summary_loss_dict_reduced)
      if(rank == 0):
        if(running_onCluster and train_iter % 200 == 0):
          now = datetime.now()
          print("{} iteration:{}".format(now, train_iter))
          print(summary_loss_dict_reduced)
          train_summary = tf_summary_from_dict(summary_loss_dict_reduced, True)
          summary_writer.add_summary(tf.compat.v1.Summary(value=train_summary), train_iter)
        elif not running_onCluster:
          train_summary = tf_summary_from_dict(summary_loss_dict_reduced, True)
          summary_writer.add_summary(tf.compat.v1.Summary(value=train_summary), train_iter)
        if train_iter % 600 == 0:
          summary_writer.flush()

      if train_iter % 1000 == 0:
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(corner_loss_dict)
        #loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * corner_weight_dict[k] for k, v in loss_dict_reduced.items() if k in corner_weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
      
        if not math.isfinite(loss_value):
            print("Corner Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            print(batch_sample_id)
            sys.exit(1)
        
        loss_dict_reduced = reduce_dict(curve_loss_dict)
        #loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * curve_weight_dict[k] for k, v in loss_dict_reduced.items() if k in curve_weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
      
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            print(batch_sample_id)
            sys.exit(1)
      t1 = time.time()
      #print("{}s elapsed for matching and compute loss".format(t1-t0))
      if(train_iter < profile_iter and perform_profile and train_iter > 2):
        profile_dict['loss_computation'].append(t1 - t0)
      
      t0 = time.time()
      optimizer.zero_grad()

      # print('tmp size:', corner_matching_indices['indices'][0].shape)
      if not args.no_topo:
        if(all_zero_corners): #if all zero exists, then add weight decay
          for params in list(corner_topo_params.values()):
            losses += 1e-8*torch.norm(params)
            #params.grad = torch.ones_like(params)
          for params in list(corner_geometry_params.values()):
            losses += 1e-8*torch.norm(params)
            #params.grad = torch.ones_like(params)
      else:
        cur_num_corner = sum([len(corner_matching_indices['indices'][i][0]) for i in range(len(corner_matching_indices['indices']))])
        cur_num_curve = sum([len(curve_matching_indices['indices'][i][0]) for i in range(len(curve_matching_indices['indices']))])
        cur_num_patch = sum([len(patch_matching_indices['indices'][i][0]) for i in range(len(patch_matching_indices['indices']))])
        # print('num corner: {} num curve: {} num patch: {}'.format(cur_num_corner, cur_num_curve, cur_num_patch))
        if cur_num_corner == 0 or cur_num_curve == 0 or cur_num_patch == 0:
        # if True:
          for params in list(corner_topo_params.values()):
            losses += 1e-8*torch.norm(params)
            #params.grad = torch.ones_like(params)
          for params in list(corner_geometry_params.values()):
            losses += 1e-8*torch.norm(params)
      
      losses.backward()
      for name, param in model_shape.named_parameters():
        if param.grad is None:
            print("param no grad", name)
        
      
      if max_norm > 0:
          if not args.clip_value:
            torch.nn.utils.clip_grad_norm_(model_shape.parameters(), max_norm)
          else:
            torch.nn.utils.clip_grad_value_(model_shape.parameters(), max_norm)
      optimizer.step()
      t1 = time.time()
      #print("{}s elapsed for compute gradients and descent".format(t1-t0))
      if(rank == 0 and train_iter < profile_iter and perform_profile and train_iter > 2):
        profile_dict['gradient_computation'].append(t1 - t0)
        if(train_iter == profile_iter - 1):
          for profile_item in profile_dict:
            print(profile_item)
            stat = np.array(profile_dict[profile_item])
            # print(stat)
            # print(stat.min(), stat.max(), stat.mean())
            print("mean: ", stat.mean())
          break
      # #for testing data
      flag_val = False
      if running_onCluster:
        # if train_iter % 300 == 0:
        if train_iter % 15000 == 0:
        # if train_iter % 15000 == 0:
        # if train_iter % 10 == 0:
          flag_val = True
      else:
        if train_iter % 10 == 0:
          flag_val = True
      if flag_val:
        # print ("validation!")
        val_data_sampler.set_epoch(val_epochs)
        val_epochs += 1
        #summary_loss owned only by gpu 0
        val_summary_loss_dict = get_val_summary_dict(model_shape, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion, val_data, device, 0, summary_loss_dict)
        val_summary_loss_dict_reduced = reduce_dict(val_summary_loss_dict)
        if rank == 0:
          test_summary = tf_summary_from_dict(val_summary_loss_dict_reduced, False) 
          # test_summary = tf_summary_from_dict(val_summary_loss_dict, False) 
          # print(test_summary)   
          summary_writer.add_summary(tf.compat.v1.Summary(value=test_summary), train_iter)

          # #for testing
          # print("val dict")
          # print(val_summary_loss_dict_reduced)
          # model_evaluation(model_without_ddp, corner_loss_criterion, curve_loss_criterion, patch_loss_criterion, val_data, device, train_iter)
          summary_writer.flush()

if __name__ == '__main__':
  if args.evalfinal or args.evaltopo or args.eval:
    eval_pipeline()
  else:
    mp.spawn(pipeline_abc,
          args=(num_of_gpus,),
          nprocs=num_of_gpus,
          join=True)