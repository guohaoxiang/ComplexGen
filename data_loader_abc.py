import numpy as np
import numba
import os
import torch

import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree as KDTree
import pickle
import random
import copy
from numpy import linalg
from methodtools import lru_cache
from torch.utils.data.distributed import DistributedSampler
import MinkowskiEngine as ME
average_patch_area = 0
average_squared_curve_length = 0
pack_size = 10000
th_norm = 1e-6
points_per_curve_dim = 34

def pack_pickle_files(data_folder, packed_data_folder):
  print("load data from {} and packing data to {}".format(data_folder, packed_data_folder))
  files = os.listdir(data_folder)
  random.shuffle(files)
  file_count = 0
  for file in files:
    if(file.endswith(".pkl")):
      print (file)
      file_count+=1
      if(file_count % pack_size == 1):
        packed_file = open(os.path.join(packed_data_folder, "packed_{:06d}.pkl".format(file_count-1)), "wb")
      with open(os.path.join(data_folder, file), "rb") as rf:
        sample = pickle.load(rf)
        sample['filename'] = file
        pickle.dump(sample, packed_file)
      if(file_count % pack_size == 0):
        packed_file.close()
  if(not packed_file.closed):
    packed_file.close()
          
def data_loader_ABC(data_folder, ignore_inpatch_curves=True):
  def normalize_model(points_with_normal, to_unit_sphere=True):
    assert(len(points_with_normal.shape) == 2 and points_with_normal.shape[1] == 6)
    points = points_with_normal[:,:3]
    normal = points_with_normal[:,3:]
    #normalize to unit bounding box
    max_coord = points.max(axis=0)
    min_coord = points.min(axis=0)
    center = (max_coord + min_coord) / 2.0
    scale = (max_coord - min_coord).max()
    normalized_points = points - center
    if(to_unit_sphere):
      scale = math.sqrt(np.square(normalized_points).sum(-1).max())*2
    # normalized_points *= 0.95/scale
    normalized_points *= 1.0/scale
    return np.concatenate([normalized_points, normal], axis=1), -center, 1.0/scale
  
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
    #update: add Extrusion and Revolution, label same as BSpline
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
  
  total_valid_corners = 0
  total_valid_curves = 0
  total_curves = 0
  total_patches = 0
  file_count = 0
  max_number_of_corners = 0
  visualize_training_samples = False
  min_number_of_curves = 1000
  min_number_points_in_patch = 10000
  max_number_points_in_patch = 0
  open_shape_counter = 0
  sample_list = []
  print("load data from {}".format(data_folder))
  if(os.path.exists(os.path.join(data_folder, "packed"))):
    print("packed pkl folder detected, will load from packed pkl file")
    read_from_packed_pkl = True
    data_folder = os.path.join(data_folder, "packed")
  else:
    read_from_packed_pkl = False
  curve_length_stat = []
  patch_area_stat = []
  
  files = os.listdir(data_folder)
  for file in files:
    if(not file.endswith(".pkl")):
      continue
    rf = open(os.path.join(data_folder, file), "rb")
    while(True):
      #load data from pkl file
      try:
        sample = pickle.load(rf)
      except EOFError:
        break
      if(read_from_packed_pkl):
        file = sample['filename']
      file_count += 1
      processed_sample = {}
      processed_sample['surface_points'] = sample['surface_points']
      scale = 1.0
      translation = np.zeros(3)
      try:
        assert(np.square(translation).sum() < 2e-6)
      except:
        print("np.square(translation).sum() = ", np.square(translation).sum())
      if processed_sample['surface_points'][:,:3].min() < -0.55 or processed_sample['surface_points'][:,:3].max() > 0.55:
        print('point cloud out of bound for model: ', file)
        continue

      assert(processed_sample['surface_points'][:,:3].min() > -0.55 and processed_sample['surface_points'][:,:3].max() < 0.55)
      if(visualize_training_samples and file_count < 100):
        np.savetxt(file.replace(".pkl", ".surface.samples.xyz"), processed_sample['surface_points'])
      
      processed_sample['curves'] = []
      total_curves += len(sample['curves'])
      corner_vert_idx = []
      corner_position = []
      
      if(len(sample['curves']) == 0):
        print("0 curves in {}".format(file))
        continue
      
      assert('inside_patch' not in sample['curves'][0])
      short_curves = False
      for curve_idx, curve in enumerate(sample['curves']):
        processed_curve = {}
        processed_curve['points'] = scale * (curve['points'] + translation)
        if (processed_curve['points'].min() < -0.55 or processed_curve['points'].max() > 0.55):
          print('min v: ', processed_curve['points'].min())
          print('max v: ', processed_curve['points'].max())
          print('file: {} curve idx: {}'.format(file, curve_idx))

        assert(processed_curve['points'].min() > -0.55 and processed_curve['points'].max() < 0.55)
        processed_curve['is_closed'] = curve['is_closed']
        if(not curve['is_closed']):
          if(not curve['start_vert_idx'] in corner_vert_idx):
            corner_vert_idx.append(curve['start_vert_idx'])
            corner_position.append(processed_curve['points'][0])
          if(not curve['end_vert_idx'] in corner_vert_idx):
            corner_vert_idx.append(curve['end_vert_idx'])
            corner_position.append(processed_curve['points'][-1])
          processed_curve['endpoints'] = [corner_vert_idx.index(curve['start_vert_idx']), corner_vert_idx.index(curve['end_vert_idx'])]
        else:
          processed_curve['endpoints'] = [-1, -1]
          
        processed_curve['type'] = curve_type_to_id(curve['type'])
        processed_curve['curve_length'] = curve['curve_length']*scale
        if(processed_curve['curve_length'] < 1e-3):
          short_curves = True
        else:
          curve_length_stat.append(processed_curve['curve_length'])
        processed_sample['curves'].append(processed_curve)
      if(short_curves):
        print("short curves detected in", file)
        continue
      processed_sample['corners'] = np.array(corner_position)
      total_valid_corners += len(processed_sample['corners'])
      total_valid_curves += len(processed_sample['curves'])
      if(len(processed_sample['curves']) < min_number_of_curves): min_number_of_curves = len(processed_sample['curves'])
      if(processed_sample['corners'].shape[0] > max_number_of_corners):
        max_number_of_corners = processed_sample['corners'].shape[0]
      if(processed_sample['corners'].shape[0] == 0):
        processed_sample['corners'] = np.zeros([0,3], dtype=np.float32)
      
      curve_occur_dict = {}
      total_patches += len(sample['patches'])
      processed_sample['patches'] = []
      for patch_idx, patch in enumerate(sample['patches']):
        processed_patch = {}
        processed_patch['type'] = patch_type_to_id(patch['type'])
        processed_patch['curves'] = patch['curves']
        for patch_curve_idx in processed_patch['curves']:
          if(patch_curve_idx in curve_occur_dict):
            curve_occur_dict[patch_curve_idx] += 1
          else:
            curve_occur_dict[patch_curve_idx] = 1
        processed_patch['patch_points'] = patch['patch_points']
        if 'grid_normal' in patch:
          processed_patch['grid_normal'] = patch['grid_normal']
        if 'u_closed' in patch:
          processed_patch['u_closed'] = patch['u_closed']
        if 'v_closed' in patch:
          processed_patch['v_closed'] = patch['v_closed']
        if(len(patch['patch_points']) < min_number_points_in_patch):
          min_number_points_in_patch = len(patch['patch_points'])
        if(len(patch['patch_points']) > max_number_points_in_patch):
          max_number_points_in_patch = len(patch['patch_points'])
        processed_patch['patch_area'] = patch['patch_area']*scale*scale
        patch_area_stat.append(processed_patch['patch_area'])
        processed_sample['patches'].append(processed_patch)
      
      for item in curve_occur_dict:
        if(curve_occur_dict[item] != 2):
          open_shape_counter += 1
          break
      
      if(visualize_training_samples and file_count < 100):
        np.savetxt(file.replace(".pkl", ".corners.xyz"), processed_sample['corners'])
        curve_positions = []
        for curve in processed_sample['curves']:
          curve_positions.append(curve['points'])
        np.savetxt(file.replace(".pkl", ".curves.samples.xyz"), np.concatenate(curve_positions, axis=0))
      processed_sample['filename'] = file
      sample_list.append(processed_sample)
    rf.close()
  print("Successfully Loaded from {} files:{}".format(file_count, len(sample_list)))
  print("max number of corners in single sample: {}".format(max_number_of_corners))
  print(min_number_of_curves, "curves at least")
  print(total_valid_curves, "valid curves total")
  print(total_valid_corners, "valid corners total")
  print(total_patches, "patches total")
  print("min and max points in single patch: {} {}".format(min_number_points_in_patch, max_number_points_in_patch))
  curve_length_stat = np.square(np.array(curve_length_stat))
  print("{} open shapes".format(open_shape_counter))
  patch_area_stat = np.array(patch_area_stat)
  print("squared curve length statistics:", len(curve_length_stat), curve_length_stat.min(), curve_length_stat.max(), curve_length_stat.mean())
  print("patch area statistics:", len(patch_area_stat), patch_area_stat.min(), patch_area_stat.max(), patch_area_stat.mean())
  global average_patch_area
  average_patch_area = patch_area_stat.mean()
  global average_squared_curve_length
  average_squared_curve_length = curve_length_stat.mean()
  return sample_list

@numba.jit()        
def points2sparse_voxel(points_with_normal, voxel_dim, feature_type, with_normal, pad1s):
    #covert to COO format, assume input points is already normalize to [-0.5 0.5]
    points = points_with_normal[:,:3] + 0.5
    voxel_dict = {}
    voxel_length = 1.0 / voxel_dim
    voxel_coord = np.clip(np.floor(points / voxel_length).astype(np.int32), 0, voxel_dim-1)
    points_normal_norm = linalg.norm(points_with_normal[:,3:], axis=1, keepdims=True)
    points_normal_norm[points_normal_norm < th_norm] = th_norm
    if(feature_type == 'local'):
      local_coord = (points - voxel_coord.astype(np.float32)*voxel_length)*voxel_dim - 0.5
      local_coord = np.concatenate([local_coord, points_with_normal[:,3:] / points_normal_norm, np.ones([local_coord.shape[0], 1])], axis=-1)
    elif(feature_type == 'global'):
      local_coord = points - 0.5
      local_coord = np.concatenate([local_coord, points_with_normal[:,3:] / points_normal_norm, np.ones([local_coord.shape[0], 1])], axis=-1)
    
    stat_voxel_dict = {}
    
    for i in range(voxel_coord.shape[0]):
      coord_tuple = (voxel_coord[i,0], voxel_coord[i,1], voxel_coord[i,2])
      if(coord_tuple not in voxel_dict):
        voxel_dict[coord_tuple] = local_coord[i]
      else:
        voxel_dict[coord_tuple] += local_coord[i]
    
    locations = np.array(list(voxel_dict.keys()))
    features = np.array(list(voxel_dict.values()))
    points_in_voxel = features[:,6:]
    features = features / points_in_voxel #pad ones
    position = features[:,:3]
    normals = features[:,3:6]
    pad_ones = features[:,6:]
    normals /= linalg.norm(normals, axis=-1, keepdims=True) + 1e-10
    
    '''
    max_variance = 0
    max_variance_signals = None
    #do the statistics
    for item in stat_voxel_dict:
      if(len(stat_voxel_dict[item]) == 1): continue
      mean_normal = voxel_dict[item][3:6]
      mean_normal /= linalg.norm(mean_normal)
      voxel_normals = np.stack(stat_voxel_dict[item], axis=0)[:,3:6]
      voxel_normals /= linalg.norm(voxel_normals, axis=1, keepdims=True)
      diff = np.square(voxel_normals - np.reshape(mean_normal, [-1, 3])).sum(-1).mean()
      if(diff > max_variance):
        max_variance = diff
        max_variance_signals = stat_voxel_dict[item]
    
    print("max variance voxel {} {}".format(max_variance, max_variance_signals))
    '''
    if(with_normal and pad1s):
      features = np.concatenate([position, normals, pad_ones], axis=1)
    elif(pad1s):
      features = np.concatenate([position, pad_ones], axis=1)
    elif(with_normal):
      features = np.concatenate([position, normals], axis=1)
    else:
      features = position
    
    return locations.astype(np.int32), features.astype(np.float32)


def points2sparse_voxel_mink(points_with_normal, voxel_dim, feature_type, with_normal, pad1s):
    """
      Use Minkowski Engine's native sparse tensorfield algorithm to voxelize the point cloud.
    """
    voxel_size = 1.0 / voxel_dim
    coords = points_with_normal[:,:3] + 0.5
    assert (feature_type == 'global')
    fea = points_with_normal[:,:3]
    if with_normal:
      points_normal_norm = linalg.norm(points_with_normal[:,3:], axis=1, keepdims=True)
      points_normal_norm[points_normal_norm < th_norm] = th_norm
      fea = np.concatenate([fea, points_with_normal[:,3:] / points_normal_norm], axis=-1)
    if pad1s:
      fea = np.concatenate([fea, np.ones([fea.shape[0], 1])], axis=-1)

    sinput = ME.SparseTensor(
      features=torch.from_numpy(fea), # Convert to a tensor
      coordinates=ME.utils.batched_coordinates([coords / voxel_size]),  # coordinates must be defined in a integer grid. If the scale
      quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE  # when used with continuous coordinates, average features in the same coordinate
    ).detach()
    return sinput.coordinates_at(0).numpy().astype(np.int32), sinput.features_at(0).numpy().astype(np.float32)


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data.astype(np.float32)

flag_normal_noise = True
r_normal_noise = 0.2

class ABCDataset(torch.utils.data.Dataset):
    def __init__(self, data, voxel_dim, feature_type='local', pad1s=True, random_rotation=False, random_angle = False, with_normal=True, flag_quick_test = False, flag_noise = 0, flag_grid = False, num_angles = 4, flag_patch_uv = False, flag_backbone = False, dim_grid = 10, eval_res_cov = False):
        self.data = data
        self.voxel_dim = voxel_dim
        self.feature_type = feature_type
        assert(self.feature_type == 'global' or self.feature_type == 'local' or self.feature_type == 'occupancy')
        self.pad1s = pad1s
        self.random_rotation_augmentation = random_rotation
        self.random_angle = random_angle
        if(self.random_rotation_augmentation): print("enable rotation augmentation")
        self.with_normal = with_normal
        self.flag_quick_test = flag_quick_test
        self.flag_noise = flag_noise
        self.flag_grid = flag_grid
        self.num_angles = num_angles
        self.flag_patch_uv = flag_patch_uv
        self.flag_backbone = flag_backbone
        self.dim_grid = dim_grid
        self.eval_res_cov = eval_res_cov
        self.fourteen_mat = []
        for i in range(4):
          self.fourteen_mat.append(R.from_rotvec(np.pi/2 * i * np.array([0,1,0])).as_matrix())
        self.fourteen_mat.append(R.from_rotvec(np.pi/2 * 1 * np.array([1,0,0])).as_matrix())
        self.fourteen_mat.append(R.from_rotvec(np.pi/2 * 3 * np.array([1,0,0])).as_matrix())
        c = np.sqrt(3)/3
        s = -np.sqrt(6)/3
        cornerrot1 = np.array([[c,0,-s],[0,1,0],[s,0,c]])
        for i in range(4):
          self.fourteen_mat.append( np.matmul(R.from_rotvec((np.pi/2 * i + np.pi / 4) * np.array([0,0,1])).as_matrix(), cornerrot1).transpose() )
        
        c = -np.sqrt(3)/3
        cornerrot2 = np.array([[c,0,-s],[0,1,0],[s,0,c]])
        for i in range(4):
          self.fourteen_mat.append( np.matmul(R.from_rotvec((np.pi/2 * i + np.pi / 4) * np.array([0,0,1])).as_matrix(), cornerrot2).transpose() )
        if(self.with_normal): print("normal is included in input signal")
        
        '''
        #compute covariance matrix in advance
        for i in range(len(self.data)):
          surface_points = self.data[i]['surface_points'][:,:3]
          points_kd_tree = KDTree(surface_points)
          one_distances, one_vertex_ids = points_kd_tree.query(surface_points, k=10)
          one_vertex_ids = np.reshape(one_vertex_ids, [-1])
          diff = np.reshape(surface_points[one_vertex_ids], [-1, 10, 3]) - np.tile(np.reshape(surface_points, [-1, 1, 3]), [1, 10, 1]) #in shape [n, k ,3]
          covariance = np.matmul(np.transpose(diff, [0, 2, 1]), diff) #in shape [n, 3, 3]
          assert(covariance.shape[0] == surface_points.shape[0])
          self.data[i]['covariance'] = covariance
        '''
    #not used
    def preprocess_data(self):
        self.processed_data = []
        for i in range(len(self.data)):
          self.processed_data.append(self.__getitem__process(i))
        if(self.random_rotation_augmentation):
          for augmentation_time in range(3):
            for i in range(len(self.data)):
              self.processed_data.append(self.__getitem__process(i))
    
    def __len__(self):
        #for testing
        return len(self.data)
        if self.flag_quick_test:
          return 10
        else:
          return len(self.data)
        if(not self.random_rotation_augmentation):
          return len(self.data)
        else:
          return 5*len(self.data)
    
    #only worked when no rotation augmentation is used
    # @lru_cache(maxsize=None)
    def __getitem__(self, idx):
        sample_data = self.data[idx % len(self.data)]
        item_points = sample_data['surface_points'].astype(np.float32) #nx6
        if self.flag_backbone:
          item_points_ori = np.copy(item_points)
        corners = sample_data['corners']
        curves = copy.deepcopy(sample_data['curves'])

        if self.flag_noise > 0:
          if self.flag_noise == 1:
            sigma=0.01
          elif self.flag_noise == 2:
            sigma =0.02
            # print('noise level 2')
          elif self.flag_noise == 3:
            sigma = 0.05 

          clip= 5.0 * sigma
          jittered_data_pts = np.clip(sigma * np.random.randn(item_points.shape[0],3), -1 * clip, clip)
          item_points[:,:3] = item_points[:,:3] + jittered_data_pts

          if flag_normal_noise:
            normal_noise = np.random.random_sample((item_points.shape[0],3)) *2 -1
            normal_noise_norm = np.linalg.norm(normal_noise, axis =-1).reshape(-1,1)
            normal_noise_norm[normal_noise_norm < th_norm] = th_norm
            normal_noise = normal_noise / normal_noise_norm
            new_normal = item_points[:,3:] + normal_noise * r_normal_noise
            new_normal_norm = np.linalg.norm(new_normal, axis = -1).reshape(-1,1)
            new_normal_norm[new_normal_norm < th_norm] = th_norm
            item_points[:, 3:] = new_normal / new_normal_norm

        patches = copy.deepcopy(sample_data['patches'])
        if self.eval_res_cov:
          for patch_idx in range(len(patches)):
            tmp_patch_pc = patches[patch_idx]['patch_points'].reshape(-1, 6)[:, :3].astype(np.float32) #not using normal
            patches[patch_idx]['patch_pc'] = tmp_patch_pc

        if(self.random_rotation_augmentation):
          if not self.random_angle:
            if self.num_angles == 4:
              rot_z = R.from_rotvec(np.pi/2 * random.randint(0,3) * np.array([0,0,1])).as_matrix()
              rot = rot_z
            elif self.num_angles == 56:
              rot = self.fourteen_mat[random.randint(0,13)]
              rot_z = R.from_rotvec(np.pi/2 * random.randint(0,3) * np.array([0,0,1])).as_matrix()
              rot = np.matmul(rot_z, rot)
            elif self.num_angles == 14:
              rot = self.fourteen_mat[random.randint(0,13)]
            elif self.num_angles == -1:
              rotation_angle = np.random.uniform() * 2 * np.pi
              cosval = np.cos(rotation_angle)
              sinval = np.sin(rotation_angle)
              rot = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
          else:
            rot = R.random().as_matrix()

          item_points = np.reshape(item_points, [-1,3])
          item_points = np.dot(item_points, rot) #apply the transform
          item_points = np.reshape(item_points, [-1,6])
          if self.flag_backbone:
            item_points_ori = np.reshape(item_points_ori, [-1,3])
            item_points_ori = np.dot(item_points_ori, rot) #apply the transform, save as matmul
            item_points_ori = np.reshape(item_points_ori, [-1,6])
          corners = np.dot(corners, rot)
          for curve in curves:
            curve['points'] = np.matmul(curve['points'], rot)
          
          if self.flag_grid:
            for patch_idx in range(len(patches)):
              tmp_patchnormal = patches[patch_idx]['grid_normal'].reshape(-1,3)
              tmp_patchnormal = np.dot(tmp_patchnormal, rot)
              patches[patch_idx]['grid_normal'] = tmp_patchnormal.reshape(-1,6).astype(np.float32)
          
          if self.eval_res_cov:
            for patch_idx in range(len(patches)):
              patches[patch_idx]['patch_pc'] = np.dot(patches[patch_idx]['patch_pc'], rot).astype(np.float32)
                        
        if not self.flag_grid:
          for patch_idx in range(len(patches)):
            #might not work for noisy case, cause the pred patch depends on the input points
            patches[patch_idx]['patch_points'] = item_points[patches[patch_idx]['patch_points']]
            assert(len(patches[patch_idx]['patch_points'].shape) == 2 and patches[patch_idx]['patch_points'].shape[1] == 6)
            
            patches[patch_idx]['patch_normals'] = patches[patch_idx]['patch_points'][:,3:].astype(np.float32)
            patch_normal_norm = np.linalg.norm(patches[patch_idx]['patch_normals'], axis = -1).reshape(-1,1)
            patch_normal_norm[patch_normal_norm < th_norm] = th_norm
            patches[patch_idx]['patch_normals'] = patches[patch_idx]['patch_normals']/ patch_normal_norm

            patches[patch_idx]['patch_points'] = patches[patch_idx]['patch_points'][:,:3].astype(np.float32)
        else:
          #rotation
          for patch_idx in range(len(patches)):
            if len(patches[patch_idx]['grid_normal']) == self.dim_grid * self.dim_grid:
              patches[patch_idx]['patch_points'] = patches[patch_idx]['grid_normal'][:,:3].astype(np.float32)
              patches[patch_idx]['patch_normals'] = patches[patch_idx]['grid_normal'][:,3:].astype(np.float32)
            else:
              #20x20
              tmp = patches[patch_idx]['grid_normal'].astype(np.float32).reshape(20, 20, -1)
              tmp = tmp[::2, ::2].reshape(-1,6)
              assert(len(tmp) == 100)
              patches[patch_idx]['patch_points'] = tmp[:,:3]
              patches[patch_idx]['patch_normals'] = tmp[:,3:]
        locations, features = points2sparse_voxel(item_points, self.voxel_dim, self.feature_type, self.with_normal, self.pad1s)
        if self.flag_backbone:
          locations_ori, features_ori = points2sparse_voxel(item_points_ori, self.voxel_dim, self.feature_type, self.with_normal, self.pad1s)
          return (locations, features, locations_ori, features_ori, sample_data['filename'])

        return (locations, features, corners.astype(np.float32), curves, patches, sample_data['filename'], item_points)


def train_data_loader(batch_size=32, voxel_dim=128, feature_type='local', pad1s=True, data_folder="/mnt/data/shilin/detr/ABC/train", rotation_augmentation=False, random_angle = False, with_normal=True, with_distribute_sampler=True, flag_quick_test = False, flag_noise = 0, flag_grid = False, num_angle = 4, flag_patch_uv = False, flag_backbone = False, dim_grid = 10, eval_res_cov = False):
    #default parameters:
    #input_feature_type: global
    #backbone_feature_encode: false
    #rotation_augment: false
    #input normal signal: false
    #with distribute sampler: true

  def pack_curve_list(curves):
    result = {}
    if(len(curves) != 0):
      labels = [curve['type'] for curve in curves]
      geometry = [np.reshape(curve['points'], [1, -1, 3]) for curve in curves]
      is_closed = [curve['is_closed'] for curve in curves]
      endpoints = [curve['endpoints'] for curve in curves]
      curve_length = [(curve['curve_length']**2) / average_squared_curve_length if (curve['curve_length']**2) / average_squared_curve_length > 0.001 else 0.001 for curve in curves]
      result['labels'] = torch.from_numpy(np.array(labels, dtype=np.int64))#.to(device)
      curve_points = torch.from_numpy(np.concatenate(geometry, axis=0).astype(np.float32))#.to(device) 
      if curve_points.shape[1] != points_per_curve_dim:
        curve_points = curve_points[:, ::3] #sample 34 points
      result['curve_points'] = curve_points
      result['is_closed'] = torch.from_numpy(np.array(is_closed, dtype=np.int64))#.to(device)
      result['endpoints'] = torch.from_numpy(np.array(endpoints, dtype=np.int64))#.to(device)
      result['curve_length_weighting'] = torch.from_numpy(1.0 / np.array(curve_length, dtype=np.float32))
      result['curve_length_weighting'] /= result['curve_length_weighting'].mean()
    else:
      result['labels'] = torch.zeros([0], dtype=torch.long)#, device=device
      result['curve_points'] = torch.zeros([0, 34, 3], dtype=torch.float32)
      result['is_closed'] = torch.zeros([0], dtype=torch.long)
      result['endpoints'] = torch.zeros([0,2], dtype=torch.long)
      result['curve_length_weighting'] = torch.zeros([0], dtype=torch.float32)
    return result
  
  def pack_patch_list(patches, n_curves):
    assert(len(patches) > 0)
    labels = torch.from_numpy(np.array([patch['type'] for patch in patches], dtype=np.int64))#.to(device)
    geometry = [torch.from_numpy(patch['patch_points']) for patch in patches] #.to(device)
    normals = [torch.from_numpy(patch['patch_normals']) for patch in patches] #.to(device)
    patch_area = [patch['patch_area'] / average_patch_area if patch['patch_area'] / average_patch_area > 0.001 else 0.001 for patch in patches]
    patch_curve_correspondence = torch.zeros([len(patches), n_curves], dtype = torch.float32)
    for patch_idx in range(len(patches)):
      patch_curve_correspondence[patch_idx, patches[patch_idx]['curves']] = 1
    result = {}
    result['labels'] = labels
    result['patch_points'] = geometry # a list
    result['patch_normals'] = normals # a list
    if eval_res_cov:
      patch_pcs = [torch.from_numpy(patch['patch_pc']) for patch in patches]
      result['patch_pcs'] = patch_pcs 

    result['patch_curve_correspondence'] = patch_curve_correspondence
    result['patch_area_weighting'] = torch.from_numpy(1.0 / np.array(patch_area, dtype=np.float32))
    result['patch_area_weighting'] /= result['patch_area_weighting'].mean()

    if flag_patch_uv or flag_grid:
      u_closed = [patch['u_closed'] for patch in patches]
      v_closed = [patch['v_closed'] for patch in patches]
      assert(not (u_closed == False and v_closed == True))
      result['u_closed'] = torch.from_numpy(np.array(u_closed, dtype=np.int64))#.to(device)\
      result['v_closed'] = torch.from_numpy(np.array(v_closed, dtype=np.int64))#.to(device)

    return result
  
  def collate_function(tensorlist):
    #first call __getitem__ then this function
    batch_size = len(tensorlist)
    locations = [np.concatenate([tensorlist[i][0], np.ones([tensorlist[i][0].shape[0], 1], dtype=np.int32)*i], axis=-1) for i in range(batch_size)]
    features = [tensorlist[i][1] for i in range(batch_size)]
    if flag_backbone:
      locations_ori = [np.concatenate([tensorlist[i][2], np.ones([tensorlist[i][2].shape[0], 1], dtype=np.int32)*i], axis=-1) for i in range(batch_size)]
      features_ori = [tensorlist[i][3] for i in range(batch_size)]
      input_sample_idx = [tensorlist[i][4] for i in range(batch_size)]
      return torch.from_numpy(np.concatenate(locations, axis=0)), torch.from_numpy(np.concatenate(features, axis=0)),\
      torch.from_numpy(np.concatenate(locations_ori, axis=0)), torch.from_numpy(np.concatenate(features_ori, axis=0)),input_sample_idx

    corner_points = [np.reshape(tensorlist[i][2], [-1,3]) for i in range(batch_size)]
    corner_batch_idx = [np.ones(tensorlist[i][2].shape[0], dtype=np.int32)*i for i in range(batch_size)]
    curves = [pack_curve_list(tensorlist[i][3]) for i in range(batch_size)]
    patches = [pack_patch_list(tensorlist[i][4], len(tensorlist[i][3])) for i in range(batch_size)]
    input_sample_idx = [tensorlist[i][5] for i in range(batch_size)]
    input_pointcloud = [tensorlist[i][6] for i in range(batch_size)] #np.stack(
    #.to(device)
    return torch.from_numpy(np.concatenate(locations, axis=0)), torch.from_numpy(np.concatenate(features, axis=0)),\
           torch.from_numpy(np.concatenate(corner_points, axis=0)), torch.from_numpy(np.concatenate(corner_batch_idx, axis=0)),\
           input_pointcloud, input_sample_idx, curves, patches, tensorlist[0][1]
  
  if not os.path.exists(os.path.join(data_folder, "packed")):
    os.mkdir(os.path.join(data_folder, "packed"))
    pack_pickle_files(data_folder, os.path.join(data_folder, "packed"))
  
  train_dataset = ABCDataset(data_loader_ABC(data_folder), voxel_dim, feature_type=feature_type, pad1s=pad1s, random_rotation=rotation_augmentation, random_angle = random_angle, with_normal=with_normal, flag_quick_test=flag_quick_test, flag_noise=flag_noise, flag_grid = flag_grid, num_angles = num_angle, flag_patch_uv = flag_patch_uv, flag_backbone = flag_backbone, dim_grid = dim_grid, eval_res_cov=eval_res_cov)
  if(with_distribute_sampler): #train mode true
    train_sampler = DistributedSampler(train_dataset)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_function, drop_last=True, sampler=train_sampler)
    return train_data, train_sampler
  else:
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_function, drop_last=True, shuffle=False, num_workers=4)
    return train_data
    