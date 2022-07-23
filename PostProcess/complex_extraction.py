from sys import base_exec_prefix
import numpy as np
import pickle
import mosek
from numpy.lib.stride_tricks import as_strided
from numpy.lib.type_check import real
from scipy.optimize import linprog
import time
import os
from scipy.sparse import csr_matrix
from scipy.sparse import csc, csc_matrix
import copy
from multiprocessing import Pool

# import numpy as np
from sklearn.neighbors import NearestNeighbors

import argparse

import json

parser = argparse.ArgumentParser()
parser.add_argument('--no_nms', action = 'store_true', help = 'using nms')
parser.add_argument('--r2', action = 'store_true', help = 'using r2')
parser.add_argument('--type', default=1, type=int, help = 'optimization type') #0: ori, 1, mix, 2, geom, 3: mul
parser.add_argument('--geom_con', action = 'store_true', help = 'adding geometric constraints')
parser.add_argument('--no_normalize', action = 'store_true', help = 'not using normalize')
parser.add_argument('--skip', action = 'store_true', help = 'skip existing files')
parser.add_argument('--geom_th', default=0.3, type=float) #used for applying geometric constraints
parser.add_argument('--th_valid', default=0.3, type=float)
parser.add_argument('--folder', type=str, required = True)
args = parser.parse_args()

flag_parallel = False
num_parallel = 20

dtype = 'float64'
th_quad_error = 0.1
th_int = 0.1 #decide whether a number is int or not
th_sol = 1e-4

th_valid = args.th_valid

th_nms_dist = 0.1 #merging threshold
th_nms_dist_patch = 0.03 #merging threshold for patch, smaller value

max_time = 599.0

d = 0.1 #curvecorner
d_patch_curve = 0.1
d_patch_corner = 0.1

def export_corners(filename, corners, idx):
  np.savetxt(filename, corners['position'][idx])

def export_curves(output_obj_file, curve_points, points_number_per_curve=34):
  curve_points = np.reshape(curve_points, [-1,3])
  with open(output_obj_file, "w") as wf:
    for point in curve_points:
      wf.write("v {} {} {}\n".format(point[0], point[1], point[2]))
    for i in range(curve_points.shape[0]):
      if(i % points_number_per_curve == (points_number_per_curve - 1)):
        continue
      wf.write("l {} {}\n".format(i+1, i+2))

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

flag_enlarge = False
enlarge_factor = 100000.0
flag_sparse = True
flag_mosek = True
# flag_int = False
flag_int = True

d = 0.2 #curvecorner
d_patch_curve = 0.1
d_patch_corner = 0.2
def get_curve_corner_similarity_geom(data, flag_exp = True):
  all_pts = data['corners']['prediction']['position']
  all_e1 = data['curves']['prediction']['points'][:,0:1,:]
  all_e2 = data['curves']['prediction']['points'][:,-1:,:]
  all_e1_diff = all_e1 - all_pts
  print('all pts shape: ', all_pts.shape)
  print('all_e1 diff shape: ', all_e1_diff.shape)
  all_e2_diff = all_e2 - all_pts
  all_e1_dist = np.linalg.norm(all_e1_diff, axis = -1)
  all_e2_dist = np.linalg.norm(all_e2_diff, axis = -1)
  all_curve_corner_dist = np.min(np.array([all_e2_dist, all_e1_dist]),axis = 0)
  # print('all_curve_corner_dist shape: ', all_curve_corner_dist.shape)
  if flag_exp:
    sim = np.exp( -all_curve_corner_dist * all_curve_corner_dist / (d * d))
  else:
    sim = all_curve_corner_dist

  return sim

def get_patch_curve_similarity_geom(data, flag_exp = True):
  all_curve_pts = data['curves']['prediction']['points']
  all_patch_pts = data['patches']['prediction']['points']
  nf = all_patch_pts.shape[0]
  nc = all_curve_pts.shape[0]
  sim = np.zeros([nf, nc])
  for i in range(nf):
    for j in range(nc):
      pts_diff = np.expand_dims(all_curve_pts[j], 1) - all_patch_pts[i]
      # print('pts diff shape: ', pts_diff.shape)
      pts_dist = np.linalg.norm(pts_diff, axis = -1)
      sim[i,j] = np.mean(pts_dist.min(-1))
  if flag_exp:
    sim = np.exp(-sim * sim / (d_patch_curve * d_patch_curve))
  
  return sim

def get_patch_corner_similarity_geom(data, flag_exp = True):
  all_corner_pts = data['corners']['prediction']['position']
  all_patch_pts = data['patches']['prediction']['points']
  nf = all_patch_pts.shape[0]
  nv = all_corner_pts.shape[0]
  sim = np.zeros([nf, nv])
  for i in range(nf):
    for j in range(nv):
      pts_diff = all_patch_pts[i] - all_corner_pts[j]
      pts_dist = np.linalg.norm(pts_diff, axis = -1)
      sim[i,j] = pts_dist.min()
  if flag_exp:
    sim = np.exp(-sim * sim / (d_patch_corner * d_patch_corner))
  return sim

w_curvecorner_topo = 0.5
w_curvecorner_geom = 0.5
flag_cgrad = True

solver = 'gurobi' #mosek, admm, gurobi
from gurobi_wrapper import solve_binary_programming_gurobi, solve_linear_programming_gurobi

flag_aux = True #using auxiliary variables
flag_gurobi_int = True
flag_only_valid = True
flag_opt_type = args.type #0: ori, 1, mix, 2, geom, 3: mul

if args.geom_con:
  flag_opt_type = 0

weight_open = 10.0 #origin version in paper

weight_unary = 10.0
weight_topo  = 10.0 #only used when normalized
# weight_topo  = 1

flag_debug = False
flag_remove_extra_constraints = True #set to true if you want to remove extra constraints

flag_normalize = False #not normalize by default.
inf = 0.0

def programming_ilp(data_input):
  #convert the program to pure integer linear programming by introducing more variables
  sample_id = data_input['sample_id']
  print("working on case {}".format(sample_id))
  
  #primitives
  corners = data_input['corners']['prediction']
  curves = data_input['curves']['prediction']
  patches = data_input['patches']['prediction']
  
  curve_corner_similarity = data_input['curve_corner_similarity'].astype(dtype)
  patch_curve_similarity = data_input['patch_curve_similarity'].astype(dtype)
  # patch_corner_similarity = (np.clip(np.matmul(patch_curve_similarity, curve_corner_similarity) / 2.0, a_min=0, a_max=1) > 0.5).astype(np.float32) #100x100
  patch_corner_similarity = data_input['patch_corner_similarity'].astype(dtype)
  # print(patch_corner_similarity.min(), patch_corner_similarity.max())
  # input()
  print('mean topo error: ', np.abs((patch_corner_similarity - np.matmul(patch_curve_similarity, curve_corner_similarity) / 2.0)).mean())
  
  # valid_corners = []
  # valid_curves = []
  # for i in range(100):
  #   if(corners['valid_prob'][i] > 0.5):
  #     valid_corners.append((i, corners['valid_prob'][i]))
  
  # for i in range(100):
  #   if(curves['valid_prob'][i] > 0.5):
  #     valid_curves.append((i, curves['valid_prob'][i]))
  
  # print("{} valid corners".format(len(valid_corners)))
  # print(valid_corners)
  # print("{} valid curves".format(len(valid_curves)))
  # print(valid_curves)
  
  export_corners("predict_corner.xyz", corners, np.where(corners['valid_prob'] > th_valid))
  export_curves("predict_curves.obj", curves['points'][np.where(curves['valid_prob'] > th_valid)])
  export_patches(patches['points'][np.where(patches['valid_prob'] > th_valid)], "predict_patches.obj")
  export_patches(patches['points'][np.where(patches['valid_prob'] > 0.0)], "predict_patches_all.obj")
  export_patches(patches['points'][np.where(patches['valid_prob'] > 0.1)], "predict_patches_01.obj")
  # export_patches(patches['points'][np.where(patches['valid_prob'] > 0.3)], "predict_patches_03.obj")
  
  # assert(curve_corner_similarity.shape == patch_curve_similarity.shape)
  # already mulified in merging step
  # for i in range(curve_corner_similarity.shape[0]): #curve_id
  #   for j in range(curve_corner_similarity.shape[1]): #corner_id
  #     curve_corner_similarity[i][j] *= corners['valid_prob'][j] * curves['valid_prob'][i] * (1 - curves['closed_prob'][i])
      
  # for i in range(patch_curve_similarity.shape[0]): #patch_id
  #   for j in range(patch_curve_similarity.shape[1]): #curve_id
  #     patch_curve_similarity[i][j] *= patches['valid_prob'][i] * curves['valid_prob'][j]
  
  # valid_curve_corner_similarity = curve_corner_similarity[np.where(curves['valid_prob'] > 0.5)][:,  np.where(corners['valid_prob'] > 0.5)]
  # print(valid_curve_corner_similarity)
  # print(valid_curve_corner_similarity.sum(axis=-1))
  
  fixed_variable_set = 'PC'
  
  corners_valid_prob = corners['valid_prob'].astype(dtype)
  curves_valid_prob = curves['valid_prob'].astype(dtype)
  patches_valid_prob = patches['valid_prob'].astype(dtype)
  open_curve_prob = 1 - curves['closed_prob'].astype(dtype)

  #enlarge numbers
  if flag_enlarge:
    corners_valid_prob = enlarge_factor * corners_valid_prob
    curves_valid_prob = enlarge_factor * corners_valid_prob
    patches_valid_prob = enlarge_factor * corners_valid_prob
    open_curve_prob = enlarge_factor * corners_valid_prob
    patch_curve_similarity = enlarge_factor * patch_curve_similarity
    curve_corner_similarity = enlarge_factor * curve_corner_similarity
    patch_corner_similarity = enlarge_factor * patch_corner_similarity
  
  nv = 100 #vertex
  nc = 100 #curve
  nf = 100 #patch

  if flag_only_valid:
    init_valid_corner_id = np.where(corners_valid_prob > th_valid)
    init_valid_curve_id = np.where(curves_valid_prob > th_valid)
    init_valid_patch_id = np.where(patches_valid_prob > th_valid)
    nv = np.where(corners_valid_prob > th_valid)[0].shape[0]
    nc = np.where(curves_valid_prob > th_valid)[0].shape[0]
    nf = np.where(patches_valid_prob > th_valid)[0].shape[0]

  w_v = 0.0
  w_c = 0.0
  w_f = 0.0
  if nv > 0:
    w_v = 1.0 / nv

  if nc > 0:
    w_c = 1.0 / nc

  if nf > 0:
    w_f = 1.0 / nf

  w_cv = 0.0
  w_fv = 0.0
  w_fc = 0.0
  if nv > 0 and nc > 0:
    w_cv = 1.0 / (nv * nc)

  if nv > 0 and nf > 0:
    w_fv = 1.0 / (nf * nv)

  if nc > 0 and nf > 0:
    w_fc = 1.0 / (nf * nc)

  print('weight: wv {} wc {} wf {} wcv {} wfv {} wfc {}'.format(w_v, w_c, w_f, w_cv, w_fv, w_fc))

  print ('number of points: {} curves: {} patches: {}'.format(nv, nc, nf))
  #variables
  # c = np.zeros(300+100+30000, dtype=dtype) #corner, curves, patch, curve-corner patch-corner patch-curve, openness
  if flag_aux:
    # c = np.zeros(300+100+30000 + 100 * 100 * 100, dtype=dtype) #curve-corner patch-corner patch-curve, aux var
    # c = np.zeros(nv + nc + nf + nc + nv * nc + nv * nf + nc * nf + nv * nc * nf, dtype=dtype) #curve-corner patch-corner patch-curve, aux var
    c = np.zeros(nv + nc + nf + nc + nv * nc + nv * nf + nc * nf + nv * nc * nf + nc, dtype=dtype) #curve-corner patch-corner patch-curve, aux var
  else:
    # c = np.zeros(300+100+30000, dtype=dtype) #curve-corner patch-corner patch-curve
    c = np.zeros(nv + nc + nf + nc + nv * nc + nv * nf + nc * nf, dtype=dtype) #curve-corner patch-corner patch-curve
    
  
  #inequalities
  # A_ub = np.zeros([60500, c.shape[0]])
  # b_ub = np.zeros(A_ub.shape[0]) #rhs
  
  # A_eq = np.zeros([10200, c.shape[0]])
  # b_eq = np.zeros(A_eq.shape[0])
  
  #redefine nv, curve-corner, patch-corner, patch-curve
  curve_corner_similarity_geom = get_curve_corner_similarity_geom(data_input)
  patch_curve_similarity_geom = get_patch_curve_similarity_geom(data_input)
  patch_corner_similarity_geom = get_patch_corner_similarity_geom(data_input)

  # np.savetxt('curvecornergeom.txt', curve_corner_similarity_geom[init_valid_curve_id][:,init_valid_corner_id].squeeze(), fmt = "%.3f")
  # np.savetxt('patchcurvegeom.txt', patch_curve_similarity_geom[init_valid_patch_id][:,init_valid_curve_id].squeeze(), fmt = "%.3f")

  # np.savetxt('patchcornergeom.txt', patch_corner_similarity_geom[init_valid_patch_id][:,init_valid_corner_id].squeeze(), fmt = "%.3f")



  #fill objective, outside
  if not flag_only_valid:
    #not updating geometry term
    c[:nv] = corners_valid_prob
    c[nv: nv + nc] = curves_valid_prob
    c[nv + nc: nv + nc + nf] = patches_valid_prob
    # c[300:10300] = np.reshape(curve_corner_similarity, [-1])
    # c[10300:20300] = np.reshape(patch_corner_similarity, [-1])
    # c[20300:30300] = np.reshape(patch_curve_similarity, [-1])
    c[nv + nc + nf: nv + nc + nf + nc * nv] = np.reshape(curve_corner_similarity, [-1])
    c[nv + nc + nf + nc * nv: nv + nc + nf + nc * nv + nf * nv] = np.reshape(patch_corner_similarity, [-1])
    c[nv + nc + nf + nc * nv + nf * nv: nv + nc + nf + nc * nv + nf * nv + nf * nc] = np.reshape(patch_curve_similarity, [-1])
    c[nv + nc + nf + nc * nv + nf * nv + nf * nc: nv + nc + nf + nc * nv + nf * nv + nf * nc + nc] = open_curve_prob*curves_valid_prob
  else:
    c[:nv] = corners_valid_prob[init_valid_corner_id]
    c[nv: nv + nc] = curves_valid_prob[init_valid_curve_id]
    c[nv + nc: nv + nc + nf] = patches_valid_prob[init_valid_patch_id]

    if flag_normalize:
      c[:nv + nc + nf] = c[:nv + nc + nf] * 2 - 1


      c[:nv] *= w_v
      c[nv: nv + nc] *= w_c
      c[nv + nc: nv + nc + nf] *= w_f
    # c[300:10300] = np.reshape(curve_corner_similarity, [-1])
    # c[10300:20300] = np.reshape(patch_corner_similarity, [-1])
    # c[20300:30300] = np.reshape(patch_curve_similarity, [-1])
    # c[nv + nc + nf: nv + nc + nf + nc * nv] = np.reshape(curve_corner_similarity[init_valid_curve_id][:,init_valid_corner_id], [-1])

    # #setting one: mix
    if flag_opt_type == 1:
      c[nv + nc + nf: nv + nc + nf + nc * nv] = np.reshape((0.5 * curve_corner_similarity + 0.5*curve_corner_similarity_geom)[init_valid_curve_id][:,init_valid_corner_id], [-1])
      c[nv + nc + nf + nc * nv: nv + nc + nf + nc * nv + nf * nv] = np.reshape(( 0.5 * patch_corner_similarity + 0.5 * patch_corner_similarity_geom)[init_valid_patch_id][:,init_valid_corner_id], [-1])
      c[nv + nc + nf + nc * nv + nf * nv: nv + nc + nf + nc * nv + nf * nv + nf * nc] = np.reshape(( 0.5 * patch_curve_similarity + 0.5 * patch_curve_similarity_geom)[init_valid_patch_id][:,init_valid_curve_id], [-1])
    elif flag_opt_type == 2:
      #setting two: geometry
      c[nv + nc + nf: nv + nc + nf + nc * nv] = np.reshape((curve_corner_similarity_geom)[init_valid_curve_id][:,init_valid_corner_id], [-1])
      c[nv + nc + nf + nc * nv: nv + nc + nf + nc * nv + nf * nv] = np.reshape((patch_corner_similarity_geom)[init_valid_patch_id][:,init_valid_corner_id], [-1])
      c[nv + nc + nf + nc * nv + nf * nv: nv + nc + nf + nc * nv + nf * nv + nf * nc] = np.reshape((patch_curve_similarity_geom)[init_valid_patch_id][:,init_valid_curve_id], [-1])

      #test0113
      # c[nv + nc + nf: nv + nc + nf + nc * nv + nf * nv + nf * nc] *= 100
    
    elif flag_opt_type == 0:
      #setting three  #mul
      # c[nv + nc + nf: nv + nc + nf + nc * nv] = np.reshape((curve_corner_similarity*curve_corner_similarity_geom)[init_valid_curve_id][:,init_valid_corner_id], [-1])
      # c[nv + nc + nf + nc * nv: nv + nc + nf + nc * nv + nf * nv] = np.reshape((patch_corner_similarity* patch_corner_similarity_geom)[init_valid_patch_id][:,init_valid_corner_id], [-1])
      # c[nv + nc + nf + nc * nv + nf * nv: nv + nc + nf + nc * nv + nf * nv + nf * nc] = np.reshape((patch_curve_similarity * patch_curve_similarity_geom)[init_valid_patch_id][:,init_valid_curve_id], [-1])
      ## c[nv + nc + nf: nv + nc + nf + nc * nv] = np.reshape((curve_corner_similarity*curve_corner_similarity_geom)[init_valid_curve_id][:,init_valid_corner_id], [-1])

      # original setting
      c[nv + nc + nf: nv + nc + nf + nc * nv] = np.reshape(curve_corner_similarity[init_valid_curve_id][:,init_valid_corner_id], [-1])
      c[nv + nc + nf + nc * nv: nv + nc + nf + nc * nv + nf * nv] = np.reshape(patch_corner_similarity[init_valid_patch_id][:,init_valid_corner_id], [-1])
      c[nv + nc + nf + nc * nv + nf * nv: nv + nc + nf + nc * nv + nf * nv + nf * nc] = np.reshape(patch_curve_similarity[init_valid_patch_id][:,init_valid_curve_id], [-1])
    elif flag_opt_type ==3:
      c[nv + nc + nf: nv + nc + nf + nc * nv] = np.reshape((curve_corner_similarity*curve_corner_similarity_geom)[init_valid_curve_id][:,init_valid_corner_id], [-1])
      c[nv + nc + nf + nc * nv: nv + nc + nf + nc * nv + nf * nv] = np.reshape((patch_corner_similarity* patch_corner_similarity_geom)[init_valid_patch_id][:,init_valid_corner_id], [-1])
      c[nv + nc + nf + nc * nv + nf * nv: nv + nc + nf + nc * nv + nf * nv + nf * nc] = np.reshape((patch_curve_similarity * patch_curve_similarity_geom)[init_valid_patch_id][:,init_valid_curve_id], [-1])
      

    # c[nv + nc + nf + nc * nv + nf * nv + nf * nc: nv + nc + nf + nc * nv + nf * nv + nf * nc + nc] = (open_curve_prob*curves_valid_prob)[init_valid_curve_id] 
    #not used since 0120
    # c[nv + nc + nf + nc * nv + nf * nv + nf * nc: nv + nc + nf + nc * nv + nf * nv + nf * nc + nc] = (open_curve_prob*curves_valid_prob)[init_valid_curve_id] * weight_open
    c[nv + nc + nf + nc * nv + nf * nv + nf * nc: nv + nc + nf + nc * nv + nf * nv + nf * nc + nc] = (open_curve_prob*curves_valid_prob)[init_valid_curve_id] 



    if flag_normalize:

      c[nv + nc + nf: nv + nc + nf + nc * nv + nf * nv + nf * nc + nc] *= 2
      c[nv + nc + nf: nv + nc + nf + nc * nv + nf * nv + nf * nc + nc] -= 1

      c[nv + nc + nf: nv + nc + nf + nc * nv] *= w_cv
      c[nv + nc + nf + nc * nv: nv + nc + nf + nc * nv + nf * nv] *= w_fv
      c[nv + nc + nf + nc * nv + nf * nv: nv + nc + nf + nc * nv + nf * nv + nf * nc] *= w_fc

      #global weight:
      c[nv + nc + nf: nv + nc + nf + nc * nv + nf * nv + nf * nc] *= weight_topo

      c[nv + nc + nf + nc * nv + nf * nv + nf * nc: nv + nc + nf + nc * nv + nf * nv + nf * nc + nc] *= w_c
  # if not flag_enlarge:
  #   c[30300:30400] = open_curve_prob*curves_valid_prob
  # else:
  #   c[30300:30400] = open_curve_prob*curves_valid_prob / enlarge_factor
  if flag_debug:
    print('!!!coeff before:', c)
  if flag_normalize:
    c = -c  #update -c
  else:
    # #all unary multliple 10
    c = 1 - c*2
    c[:nv + nc + nf] *=weight_unary
    c[nv + nc + nf + nc * nv + nf * nv + nf * nc: nv + nc + nf + nc * nv + nf * nv + nf * nc + nc] *= weight_open

  if flag_debug:
    print('!!!coeff after:', c)
  
  max_inner_iter = 1
  float_iter = 4
  min_int_iter = 2

  #set A_ub before
  # A_ub = []
  b_ub = []
  b_lb = []
  bkc = []
  x_lb = []
  x_ub = []
  bkx = []
  
  # b_ub = []
  # A_eq = []
  # b_eq = []
  # buc = []
  rows = []
  cols = []
  data = []

  cur_row = 0
  #comment on 1211
  for i in range(nc): #curve
    for j in range(nv): #corner
      #EV(i, j) < E(i)
      # oneline = np.zeros(c.shape[0], dtype=dtype)
      # # A_ub[cur_row][100+i] = -1
      # # A_ub[cur_row][300+100*i+j] = 1
      # oneline[100 + i] = -1
      # oneline[300 + 100*i + j] = 1
      # A_ub.append(oneline)
      
      if not flag_remove_extra_constraints:
        b_ub.append(0)
        b_lb.append(-inf)
        rows.append(cur_row)
        cols.append(nv + i)
        data.append(-1)
        rows.append(cur_row)
        cols.append(nv + nc + nf + nv*i + j)
        data.append(1)

        bkc.append(mosek.boundkey.up)
        cur_row += 1
      
      #EV(i, j) < V(j)
      # A_ub[cur_row][j] = -1
      # A_ub[cur_row][300+100*i+j] = 1

      # oneline = np.zeros(c.shape[0], dtype=dtype)
      # oneline[j] = -1
      # oneline[300 + 100*i + j] = 1
      # A_ub.append(oneline)

      b_ub.append(0)
      b_lb.append(-inf)
      rows.append(cur_row)
      cols.append(j)
      data.append(-1)
      rows.append(cur_row)
      cols.append(nv + nc + nf + nv*i + j)
      data.append(1)
      bkc.append(mosek.boundkey.up)
      cur_row += 1
  
  for i in range(nf): #patch
    for j in range(nc): #curve
      #FE(i,j) < F(i)
      # A_ub[cur_row][200+i] = -1
      # A_ub[cur_row][20300+100*i+j] = 1

      # oneline = np.zeros(c.shape[0], dtype=dtype)
      # oneline[200 + i] = -1
      # oneline[20300+100*i+j] = 1
      # A_ub.append(oneline)

      b_ub.append(0)
      b_lb.append(-inf)
      rows.append(cur_row)
      cols.append(nv + nc + i)
      data.append(-1)
      rows.append(cur_row)
      cols.append(nv + nc + nf + nc * nv + nf * nv +nc*i+j)
      data.append(1)
      bkc.append(mosek.boundkey.up)

      cur_row += 1
      
      #FE(i,j) < E(j) # omit cause equation 1
      # A_ub[cur_row][100+j] = -1
      # A_ub[cur_row][20300+100*i+j] = 1
      # oneline = np.zeros(c.shape[0], dtype=np.float32)
      # oneline[100 + j] = -1
      # oneline[20300+100*i+j] = 1
      # A_ub.append(oneline)
      # cur_row += 1
  
  if not flag_remove_extra_constraints:
    for i in range(nf): #patch
      for j in range(nv): #corner
        #FV(i,j) < F(i)
        # A_ub[cur_row][200+i] = -1
        # A_ub[cur_row][10300+100*i+j] = 1
        # oneline = np.zeros(c.shape[0], dtype=dtype)
        # oneline[200+i] = -1
        # oneline[10300+100*i+j] = 1
        # A_ub.append(oneline)

        b_ub.append(0)
        b_lb.append(-inf)
        rows.append(cur_row)
        cols.append(nv + nc + i)
        data.append(-1)
        rows.append(cur_row)
        # cols.append(10300  +nv*i+j)
        cols.append(nv + nc + nf + nc * nv +nv*i+j)
        data.append(1)
        bkc.append(mosek.boundkey.up)


        cur_row += 1
        #FV(i,j) < V(j)
        # A_ub[cur_row][j] = -1
        # A_ub[cur_row][10300+100*i+j] = 1
        # oneline = np.zeros(c.shape[0], dtype=dtype)
        # oneline[j] = -1
        # oneline[10300+100*i+j] = 1
        # A_ub.append(oneline)
        b_ub.append(0)
        b_lb.append(-inf)
        rows.append(cur_row)
        cols.append(j)
        data.append(-1)
        rows.append(cur_row)
        # cols.append(10300+100*i+j)
        # cols.append(10300+100*i+j)
        cols.append(nv + nc + nf + nc * nv +nv*i+j)

        data.append(1)
        bkc.append(mosek.boundkey.up)


        cur_row += 1
  
  for i in range(nv): #corner
    # A_ub[cur_row][i] = 1
    #V(i) < sigma EV
    # oneline = np.zeros(c.shape[0], dtype=dtype)
    # oneline[i] = 1
    # for j in range(100): #curve
    #   # A_ub[cur_row][300+100*j+i] = -1
    #   oneline[300 + 100 * j + i] = -1
    # A_ub.append(oneline)

    b_ub.append(0)
    b_lb.append(-inf)
    rows.append(cur_row)
    cols.append(i)
    data.append(1)
    for j in range(nc):
      rows.append(cur_row)
      cols.append(nv + nc + nf + nv * j + i)
      data.append(-1)
    bkc.append(mosek.boundkey.up)
    cur_row += 1
  
  #this constraint only work when nc > 0
  if nc > 0:
    for i in range(nf): #patch
      # A_ub[cur_row][200+i] = 1
      # oneline = np.zeros(c.shape[0], dtype=dtype)
      # oneline[200 + i] = 1
      # #F(i) < sigma FE
      # for j in range(100): #curve
      #   # A_ub[cur_row][20300+100*i+j] = -1
      #   oneline[20300+100*i+j] = -1
      # A_ub.append(oneline)

      b_ub.append(0)
      b_lb.append(-inf)
      rows.append(cur_row)
      cols.append(nv + nc + i)
      data.append(1)
      for j in range(nc):
        rows.append(cur_row)
        # cols.append(20300+nc*i+j)
        cols.append(nv + nc + nf + nc * nv + nf * nv +nc*i+j)
        data.append(-1)
      bkc.append(mosek.boundkey.up)
      cur_row += 1
  
  #comment on 1211
  if not flag_remove_extra_constraints:
    for i in range(nv): #corner
      # A_ub[cur_row][i] = 1
      # oneline = np.zeros(c.shape[0], dtype=dtype)
      # oneline[i] = 1
      # #V(i) <= sigma FV
      # for j in range(100): #patch
      #   # A_ub[cur_row][10300+100*j+i] = -1
      #   oneline[10300+100*j+i] = -1
      # A_ub.append(oneline)

      b_ub.append(0)
      b_lb.append(-inf)
      rows.append(cur_row)
      cols.append(i)
      data.append(1)
      for j in range(nf):
        rows.append(cur_row)
        # cols.append(10300+100*j+i)
        cols.append(nv + nc + nf + nc * nv+nv*j+i)

        data.append(-1)
      bkc.append(mosek.boundkey.up)

      cur_row += 1
  
  
  #removed since 0114
  #curve open_prob < valid_prob
  # for i in range(nc):
  #   #O < E
  #   # A_ub[cur_row][100+i] = -1
  #   # A_ub[cur_row][30300+i] = 1
  #   # oneline = np.zeros(c.shape[0], dtype=dtype)
  #   # oneline[100+i] = -1
  #   # oneline[30300+i] = 1
  #   # A_ub.append(oneline)

  #   b_ub.append(0)
  #   b_lb.append(-inf)
  #   rows.append(cur_row)
  #   cols.append(nv+i)
  #   data.append(-1)
  #   rows.append(cur_row)
  #   cols.append(nv + nc + nf + nc * nv + nf * nv + nf * nc+i)
  #   data.append(1)
  #   bkc.append(mosek.boundkey.up)

  #   cur_row += 1
  
  #add on 0114
  for i in range(nc):
    #Y(i) <= E_i
    b_ub.append(0)
    b_lb.append(-inf)
    rows.append(cur_row)
    cols.append(nv+i)
    data.append(-1)
    rows.append(cur_row)
    cols.append(nv + nc + nf + nc + nv * nc + nv * nf + nc * nf + nv * nc * nf + i) #y(i)
    data.append(1)
    bkc.append(mosek.boundkey.up)
    cur_row += 1

    #Y(i) <= O(i)
    b_ub.append(0)
    b_lb.append(-inf)
    rows.append(cur_row)
    # cols.append(nv+i)
    cols.append(nv + nc + nf + nc * nv + nf * nv + nf * nc+i) #O(i)
    data.append(-1)
    rows.append(cur_row)
    cols.append(nv + nc + nf + nc + nv * nc + nv * nf + nc * nf + nv * nc * nf + i) #y(i)
    data.append(1)
    bkc.append(mosek.boundkey.up)
    cur_row += 1

    #Y(i) >= E(i) + O(i) - 1

    b_ub.append(1)
    b_lb.append(-inf)
    rows.append(cur_row)
    # cols.append(nv+i)
    cols.append(nv + nc + nf + nc * nv + nf * nv + nf * nc+i) #O(i)
    data.append(1)
    rows.append(cur_row)
    # cols.append(nv+i)
    # cols.append(nv + nc + nf + nc * nv + nf * nv + nf * nc+i) #O(i)
    cols.append(nv+i) #E(i)
    data.append(1)
    
    rows.append(cur_row)
    cols.append(nv + nc + nf + nc + nv * nc + nv * nf + nc * nf + nv * nc * nf + i) #y(i)
    data.append(-1)
    bkc.append(mosek.boundkey.up)
    cur_row += 1
    

  
  #for auxiliry variables
  if flag_aux:
    for i in range(nf):
      for j in range(nc):
        for k in range(nv):
          # oneline = np.zeros(c.shape[0], dtype=dtype)
          # oneline[30400 + 100 * 100 * i + 100 * j + k] = 2
          # oneline[400 + 20000 + 100 * i + j] = -1
          # oneline[400 + 100 * j + k] = -1
          # A_ub.append(oneline)

          # #original version
          # b_ub.append(0)
          # b_lb.append(-inf)
          # rows.append(cur_row)
          # # cols.append(30400 + 100 * 100 * i + 100 * j + k)
          # # cols.append(nv + nc + nf + nc * nv + nf * nv + nf * nc + nc * nv * i + nv * j + k)
          # cols.append(nv + nc + nf + nc * nv + nf * nv + nf * nc + nc + nc * nv * i + nv * j + k)
          # data.append(2)
          # rows.append(cur_row)
          # # cols.append(400 + 20000 + 100 * i + j) #this is wrong
          # cols.append(nv + nc + nf + nv * nc + nf * nv + nc * i + j) #FE(i,j)
          
          # data.append(-1)
          # rows.append(cur_row)
          # # cols.append(400 + 100 * j + k)
          # cols.append(nv + nc + nf + nv * j + k) #EV(j,k)

          # data.append(-1)
          # bkc.append(mosek.boundkey.up)
          # cur_row += 1
          
          #modified version
          #z(i,j,k) <= FE(i,j)
          b_ub.append(0)
          b_lb.append(-inf)
          rows.append(cur_row)
          # cols.append(30400 + 100 * 100 * i + 100 * j + k)
          # cols.append(nv + nc + nf + nc * nv + nf * nv + nf * nc + nc * nv * i + nv * j + k)
          cols.append(nv + nc + nf + nc * nv + nf * nv + nf * nc + nc + nc * nv * i + nv * j + k)
          data.append(1)
          rows.append(cur_row)
          # cols.append(400 + 20000 + 100 * i + j) #this is wrong
          cols.append(nv + nc + nf + nv * nc + nf * nv + nc * i + j) #FE(i,j)
          
          data.append(-1)
          bkc.append(mosek.boundkey.up)
          cur_row += 1

          #z(i,j.k) <= EV(j,k)
          b_ub.append(0)
          b_lb.append(-inf)
          rows.append(cur_row)

          cols.append(nv + nc + nf + nc * nv + nf * nv + nf * nc + nc + nc * nv * i + nv * j + k)
          data.append(1)
          # cols.append(400 + 100 * j + k)
          rows.append(cur_row)
          cols.append(nv + nc + nf + nv * j + k) #EV(j,k)

          data.append(-1)
          bkc.append(mosek.boundkey.up)
          cur_row += 1
          # oneline = np.zeros(c.shape[0], dtype=dtype)
          # oneline[30400 + 100 * 100 * i + 100 * j + k] = -1
          # oneline[400 + 20000 + 100 * i + j] = 1
          # oneline[400 + 100 * j + k] = 1
          # A_ub.append(oneline)

          b_ub.append(1)
          b_lb.append(-inf)
          rows.append(cur_row)
          # cols.append(30400 + 100 * 100 * i + 100 * j + k)
          cols.append(nv + nc + nf + nc * nv + nf * nv + nf * nc + nc + nc * nv * i + nv * j + k)
          data.append(-1)
          rows.append(cur_row)
          # cols.append(400 + 20000 + 100 * i + j)
          cols.append(nv + nc + nf + nv * nc + nf * nv + nc * i + j) #FE(i,j)
          data.append(1)
          rows.append(cur_row)
          # cols.append(400 + 100 * j + k)
          cols.append(nv + nc + nf + nv * j + k) #EV(j,k)
          data.append(1)
          bkc.append(mosek.boundkey.up)
          cur_row += 1

  
  if args.geom_con:
    curve_corner_con = get_curve_corner_similarity_geom(data_input, False)
    patch_curve_con = get_patch_curve_similarity_geom(data_input, False)
    patch_corner_con = get_patch_corner_similarity_geom(data_input, False)
    curve_corner_con = (curve_corner_con[init_valid_curve_id][:,init_valid_corner_id] < args.geom_th).astype(dtype).reshape(nc, nv)
    patch_curve_con = (patch_curve_con[init_valid_patch_id][:,init_valid_curve_id] < args.geom_th).astype(dtype).reshape(nf, nc)
    patch_corner_con = (patch_corner_con[init_valid_patch_id][:,init_valid_corner_id] < args.geom_th).astype(dtype).reshape(nf, nv)
    #curve corner
    #for testing
    # np.savetxt('patchcurve.txt', patch_curve_con, fmt = '%d')
    # np.savetxt('patchcorner.txt', patch_corner_con, fmt = '%d')
    # np.savetxt('curvecorner.txt', curve_corner_con, fmt = '%d')
    
    #out output:
    # print('curve corner con:')
    # for i in range(nc):
    #   print('{}: {}'.format(i, np.where(curve_corner_con[i] == 1)))

    # print('patch curve con:')
    # for i in range(nf):
    #   print('{}: {}'.format(i, np.where(patch_curve_con[i] == 1)))

    # print('patch corner con:')
    # for i in range(nf):
    #   print('{}: {}'.format(i, np.where(patch_corner_con[i] == 1)))


    for i in range(nc):
      for j in range(nv):
          b_ub.append(curve_corner_con[i][j])
          b_lb.append(-inf)
          rows.append(cur_row)
          # cols.append(nv + nc + nf + nc + nv * nc + nv * nf + nc * nf + nv * nc * nf + i) #y(i)
          cols.append(nv + nc + nf + nv * i + j) #y(i)
          data.append(1)
          bkc.append(mosek.boundkey.up)
          cur_row += 1
    #patch corner
    for i in range(nf):
      for j in range(nv):
        b_ub.append(patch_corner_con[i][j])
        b_lb.append(-inf)
        rows.append(cur_row)
        # cols.append(nv + nc + nf + nc + nv * nc + nv * nf + nc * nf + nv * nc * nf + i) #y(i)
        # cols.append(nv + nc + nf + nc * i + j) #y(i)
        cols.append(nv + nc + nf + nc * nv + nv * i + j)
        data.append(1)
        bkc.append(mosek.boundkey.up)
        cur_row += 1

    #patch curve
    for i in range(nf):
      for j in range(nc):
        b_ub.append(patch_curve_con[i][j])
        b_lb.append(-inf)
        rows.append(cur_row)
        # cols.append(nv + nc + nf + nc + nv * nc + nv * nf + nc * nf + nv * nc * nf + i) #y(i)
        # cols.append(nv + nc + nf + nc * i + j) #y(i)
        cols.append(nv + nc + nf + nc * nv + nv * nf + nc * i + j)
        data.append(1)
        bkc.append(mosek.boundkey.up)
        cur_row += 1


  # A_ub = np.array(A_ub)
  # b_ub = np.zeros(A_ub.shape[0], dtype=dtype)
  # print("A ub shape: ", A_ub.shape)
  # assert(cur_row == A_ub.shape[0])
  
  #set part of A_eq
  # cur_row = 0
  #curve patch constraints (every curve has 2 patch)
  num_fix_ent = 0
  for i in range(nc): #curve
    # A_eq[cur_row][100+i] = 2
    # oneline = np.zeros(c.shape[0], dtype=dtype)
    # oneline[100 + i] = 2
    # for j in range(100): #patch
    #   # A_eq[cur_row][20300+100*j+i] = -1
    #   oneline[20300+100*j+i] = -1
    # A_eq.append(oneline)
    # b_eq.append(0.0)
    b_ub.append(0)
    b_lb.append(0)
    rows.append(cur_row)
    cols.append(nv+i)
    data.append(2)
    num_fix_ent = num_fix_ent + 1
    for j in range(nf): #patch
      rows.append(cur_row)
      cols.append(nv + nc + nf + nc * nv + nf * nv+nc*j+i)
      data.append(-1)
      num_fix_ent = num_fix_ent + 1
    bkc.append(mosek.boundkey.fx)

    cur_row += 1
  
  #curve - 2 corner constraints
  for i in range(nc): #curve
    # A_eq[cur_row][30300+i] = 2 #no validness
    # oneline = np.zeros(c.shape[0], dtype=dtype)
    # oneline[30300+i] = 2 #no validness
    # for j in range(100): #corner
    #   # A_eq[cur_row][300+ 100*i +j] = -1
    #   oneline[300+ 100*i +j] = -1
    # A_eq.append(oneline)
    # b_eq.append(0.0)

    b_ub.append(0)
    b_lb.append(0)
    rows.append(cur_row)
    # cols.append(30300+i)
    # cols.append(nv +nc + nf + nc * nv + nf * nv + nf * nc+i)
    #update 0114, replace O with Y
    cols.append(nv + nc + nf + nc + nv * nc + nv * nf + nc * nf + nv * nc * nf + i) #y(i)
    data.append(2)
    num_fix_ent = num_fix_ent + 1
    for j in range(nv): #corner
      rows.append(cur_row)
      # cols.append(300+ 100*i +j)
      cols.append(nv + nc + nf+ nv*i +j)

      data.append(-1)
      num_fix_ent = num_fix_ent + 1
    bkc.append(mosek.boundkey.fx)

    cur_row += 1

  #bnb
  opt_x = np.zeros(c.shape[0])
  obj_ub = 10.0 #upper bound

  #outer iter
  float_sol = c
  #try twice: first free all variables, then fix patch_curve_similarity
  # while outer_iter_count < 2:
  if True:
    #reset pc, cc, pc similarity
    curve_corner_similarity_tmp = curve_corner_similarity
    patch_curve_similarity_tmp = patch_curve_similarity
    patch_corner_similarity_tmp = patch_corner_similarity
    
    # A_eq_out = copy.deepcopy(A_eq)
    # b_eq_out = copy.deepcopy(b_eq)
    flag_topo_valid = False
    #inner iter
    for iter in range(max_inner_iter):
      print("iter", iter)
      assert(fixed_variable_set == 'CC' or fixed_variable_set == 'PC')
      #constraints
      # A_ub[:] = 0
      # b_ub[:] = 0
      #equalities
      # A_eq[:] = 0
      # b_eq[:] = 0
      #FE*EC = 2 *FC
      # A_eq_copy = copy.deepcopy(A_eq_out)
      # b_eq_copy = copy.deepcopy(b_eq_out)

      #patch corner
      if flag_aux:
        for i in range(nf): #patch
          for j in range(nv): #corner
            # A_eq[cur_row][10300 + 100*i + j] = -2
            # oneline = np.zeros(c.shape[0], dtype=dtype)
            # oneline[10300 + 100*i + j] = -2
            # for k in range(100): #curve
            #   #2 * patch_corner[i][j] = \sum patch_curve[i][k] * curve_corner[k][j]
            #   # sigma Z(i,j,k) over j
            #   oneline[30400 + 10000 * i + 100 * k + j] = 1
            # A_eq_copy.append(oneline)
            # b_eq_copy.append(0.0)
            b_ub.append(0)
            b_lb.append(0)
            rows.append(cur_row)
            # cols.append(10300 + 100*i + j)
            cols.append(nv + nc + nf + nc * nv + nv*i + j)
            data.append(-2)
            num_fix_ent = num_fix_ent + 1
            for k in range(nc): #curve
              rows.append(cur_row)
              # cols.append(30400 + 10000 * i + 100 * k + j) #wrong
              # cols.append(30400 + 10000 * i + 100 * k + j)
              cols.append(nv + nc + nf + nc * nv + nf * nv + nf * nc + nc + nc * nv * i + nv * k + j)
              data.append(1)
              num_fix_ent = num_fix_ent + 1
            bkc.append(mosek.boundkey.fx)
            cur_row += 1
      

      # A_eq_copy = np.array(A_eq_copy)
      # b_eq_copy = np.array(b_eq_copy)
      # assert(cur_row == A_eq.shape[0])
      # b_eq = np.zeros(A_eq_copy.shape[0], dtype=dtype)
      
      print("Solving Linear Programming")
      t0 = time.time()
      # res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(0,1), method='highs-ipm', options={'disp':True, 'dual_feasibility_tolerance':1e-16, 'primal_feasibility_tolerance':1e-16})
      
      #for model 06
      # res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(0,1), method='highs-ipm', options={'disp':True, 'dual_feasibility_tolerance':1e-7, 'primal_feasibility_tolerance':1.25e-8})
      if solver == 'gurobi':
        k2v = {"fr": 3,"fx": 2,"lo":0,"ra":4,"up":1}
        bkc_gurobi = []
        for i in range(len(bkc)):
          # print(k2v[str(repr(bkx[i]))], " ")
          bkc_gurobi.append(k2v[str(repr(bkc[i]))])
        A_sparse = csr_matrix((data, (rows, cols)), shape=(cur_row, c.shape[0]))
        if flag_debug:
          print('A sparse: ', A_sparse)
          print('c: ', c)
          print('b_lb: ',b_lb)
          print('b_ub: ', b_ub)
          print('bkc_gurobi: ', bkc_gurobi)

        if flag_gurobi_int:
          (flag_opt_fea, x) = solve_binary_programming_gurobi(c, A_sparse, b_lb, b_ub, bkc_gurobi, flag_max = False)
        else:
          (flag_opt_fea, x) = solve_linear_programming_gurobi(c, A_sparse,np.zeros(c.shape[0]), np.ones(c.shape[0]) ,b_lb, b_ub, bkc_gurobi, flag_max = False)
      # if iter < float_iter:
      #   (flag_opt_fea, x) = mosek_linprog(c, A_ub, b_ub, A_eq_copy, b_eq_copy, xbd, xbdtype, flag_max = False, flag_int = False)
      # else:
      #   #int solver with solution guess
      #   (flag_opt_fea, x) = mosek_linprog(c, A_ub, b_ub, A_eq_copy, b_eq_copy, xbd, xbdtype, flag_max = False, flag_int = True, flag_initial_guess = True, initial_guess = float_sol)

      t1 = time.time()
      diff_t = t1 - t0
      f = open(data_input['name'][:-4] + '_timing.txt', 'w')
      f.write(str(diff_t))
      f.close()

      if diff_t > max_time:
        f = open(data_input['name'] + 'maxtime', 'w')
        f.close()
      assert(flag_opt_fea)

      if flag_opt_fea:
        # if iter == float_iter - 1:
        #   float_sol = x

        if flag_enlarge:
          x = x / enlarge_factor

        # print(x.min(), x.max())
        # assert(x.shape[0] == c.shape[0])
        print((t1-t0), "s elapsed in iteration", iter)
        curve_corner_similarity_tmp = np.reshape(x[nv + nc + nf: nv + nc + nf + nc * nv], [nc, nv])    
        patch_curve_similarity_tmp = np.reshape(x[ nv + nc + nf + nc * nv + nf * nv: nv + nc + nf + nc * nv + nf * nv + nf * nc], [nf, nc])
        patch_corner_similarity_tmp = np.reshape(x[nv + nc + nf + nc * nv:nv + nc + nf + nc * nv + nf * nv], [nf,nv])
        # if iter >= float_iter + min_int_iter:
        if True:
          #corners_valid_prob = x[:100]
          #corners_valid_prob = x[:100]
          if nv != 0:
            max_topo_error = np.abs((patch_corner_similarity_tmp - np.matmul(patch_curve_similarity_tmp, curve_corner_similarity_tmp) / 2.0)).max()
            print('max topo error for iter {}: {}'.format(iter, max_topo_error))
            if max_topo_error < th_quad_error:
            # if max_topo_error < th_quad_error:
              flag_topo_valid = True
              break
        
        if fixed_variable_set == 'CC':
          fixed_variable_set = 'PC'
        else:
          fixed_variable_set = 'CC'
      else:
        x = None
        break
    
    flag_topo_valid = True
    assert(flag_topo_valid)
    if flag_topo_valid:
      #udpate solution and conf
      opt_x = x
  #output to be changed
  # assert(opt_x != None)
  print('rounding error: ', np.abs(opt_x - np.round(opt_x)).max())
  print('valid max: ', opt_x[:nf + nc + nv].max())
  print('overall max: ', opt_x.max())
  # np.savetxt('solution.txt', opt_x)
  flag_valid_opt = opt_x[:nf + nc + nv].max() > 0.5
  if not flag_valid_opt:
    return False
  assert(opt_x.max() > 0.5)
  if not flag_only_valid:
    corners_valid_prob = opt_x[:nv]
    curves_valid_prob = opt_x[nv:nv + nc]
    patches_valid_prob = opt_x[nv + nc :nv + nc + nf]
    curve_corner_similarity = curve_corner_similarity_tmp
    patch_curve_similarity = patch_curve_similarity_tmp
    patch_corner_similarity = patch_corner_similarity_tmp
    open_curve_prob = opt_x[nv + nc + nf + nc * nv + nf * nv + nf * nc: nv + nc + nf + nc * nv + nf * nv + nf * nc + nc]
  else:
    corners_valid_prob[init_valid_corner_id] = opt_x[:nv]
    curves_valid_prob[init_valid_curve_id] = opt_x[nv: nv + nc]
    patches_valid_prob[init_valid_patch_id] = opt_x[nv + nc : nv + nc + nf]
    # curve_corner_similarity[init_valid_curve_id][:,init_valid_corner_id] = np.expand_dims(curve_corner_similarity_tmp, 1)
    # patch_curve_similarity[init_valid_patch_id][:,init_valid_curve_id] = np.expand_dims(patch_curve_similarity_tmp, 1)
    # patch_corner_similarity[init_valid_patch_id][:, init_valid_corner_id] = np.expand_dims(patch_corner_similarity_tmp, 1)

    #replace the 3 equals above
    for i in range(init_valid_curve_id[0].shape[0]):
      for j in range(init_valid_corner_id[0].shape[0]):
        curve_corner_similarity[init_valid_curve_id[0][i],init_valid_corner_id[0][j]] = curve_corner_similarity_tmp[i,j]
    
    for i in range(init_valid_patch_id[0].shape[0]):
      for j in range(init_valid_corner_id[0].shape[0]):
        patch_corner_similarity[init_valid_patch_id[0][i], init_valid_corner_id[0][j]] = patch_corner_similarity_tmp[i,j]

    for i in range(init_valid_patch_id[0].shape[0]):
      for j in range(init_valid_curve_id[0].shape[0]):
        patch_curve_similarity[init_valid_patch_id[0][i], init_valid_curve_id[0][j]] = patch_curve_similarity_tmp[i,j]
    
    open_curve_prob[init_valid_curve_id] = opt_x[nv + nc + nf + nc * nv + nf * nv + nf * nc: nv + nc + nf + nc * nv + nf * nv + nf * nc + nc]

    # np.savetxt('curvecorner_1.txt', curve_corner_similarity[init_valid_curve_id][:,init_valid_corner_id].squeeze())


    # np.savetxt('curvecorner.txt', curve_corner_similarity_tmp)
    # print('curve corner sum: ', np.sum(curve_corner_similarity_tmp, -1))
    # print('openness:', opt_x[nv + nc + nf + nc * nv + nf * nv + nf * nc: nv + nc + nf + nc * nv + nf * nv + nf * nc + nc])
    # print('diff: ', np.sum(curve_corner_similarity_tmp, -1) - 2 * opt_x[nv + nc + nf + nc * nv + nf * nv + nf * nc: nv + nc + nf + nc * nv + nf * nv + nf * nc + nc])
  #update similarity
  # curve_corner_similarity = np.reshape(opt_x[300:10300], [100, 100])    
  # patch_curve_similarity = np.reshape(opt_x[20300:30300], [100, 100])
  # patch_corner_similarity = np.reshape(opt_x[10300:20300], [100,100])

  
  #valid_patch_curve = np.where(patch_curve_similarity > 0.5)
  #valid_curve_corner = np.where(patch_curve_similarity > 0.5)
  export_corners("opt_corner.xyz", corners, np.where(corners_valid_prob > th_valid))
  export_curves("opt_curves.obj", curves['points'][np.where(curves_valid_prob > th_valid)])
  export_patches(patches['points'][np.where(patches_valid_prob > th_valid)], "opt_patches.obj")
  # print(curve_corner_similarity[52][np.where(corners_valid_prob > 0.5)])
  # print(curve_corner_similarity[52].sum())
  
  # print(patch_curve_similarity[:, 52])
  
  # valid_curve_corner_similarity = curve_corner_similarity[np.where(curves_valid_prob > 0.5)][:,  np.where(corners_valid_prob > 0.5)]
  # # print(valid_curve_corner_similarity)
  # print("curve_corner info")
  # print(valid_curve_corner_similarity.sum(axis=-1))
  
  # valid_patch_corner_similarity = np.reshape(x[10300:20300], [100, 100])[np.where(patches_valid_prob > 0.5)][:, np.where(corners_valid_prob > 0.5)]
  # print(valid_patch_corner_similarity)

  # print('mean closed error: ', np.abs(patch_corner_similarity - np.matmul(patch_curve_similarity, curve_corner_similarity) / 2.0).mean())
  # print('max closed error: ', np.abs(patch_corner_similarity - np.matmul(patch_curve_similarity, curve_corner_similarity) / 2.0).max())

  
  # print('valid open curve')
  # print(open_curve_prob[np.where(curves_valid_prob > 0.5)])
  # print(open_curve_prob[np.where(curves_valid_prob > 0.5)].shape)
  data_input['curve_corner_similarity'] = curve_corner_similarity
  data_input['patch_curve_similarity'] = patch_curve_similarity
  data_input['patch_corner_similarity'] = patch_corner_similarity
  #update open closeness
  curves['closed_prob'] = 1 - open_curve_prob
  corners['valid_prob'] = corners_valid_prob
  curves['valid_prob'] = curves_valid_prob
  patches['valid_prob'] = patches_valid_prob
  
  return True
  #curve corner

def merge_duplicated_primitives(data, flag_merge = True):
  sample_id = data['sample_id']
  print("working on case {}".format(sample_id))
  
  #primitives
  corners = data['corners']['prediction']
  curves = data['curves']['prediction']
  patches = data['patches']['prediction']
  
  curve_corner_similarity = data['curve_corner_similarity']
  patch_curve_similarity = data['patch_curve_similarity']
  patch_corner_similarity = data['patch_corner_similarity'].astype(dtype)
  
  #data modified here
  assert(curve_corner_similarity.shape == patch_curve_similarity.shape)
  for i in range(curve_corner_similarity.shape[0]): #curve_id
    for j in range(curve_corner_similarity.shape[1]): #corner_id
      curve_corner_similarity[i][j] *= corners['valid_prob'][j] * curves['valid_prob'][i] * (1 - curves['closed_prob'][i])
      
  for i in range(patch_curve_similarity.shape[0]): #patch_id
    for j in range(patch_curve_similarity.shape[1]): #curve_id
      patch_curve_similarity[i][j] *= patches['valid_prob'][i] * curves['valid_prob'][j]
  

  for i in range(patch_corner_similarity.shape[0]):
    for j in range(patch_corner_similarity.shape[1]):
      patch_corner_similarity[i][j] *= patches['valid_prob'][i] * corners['valid_prob'][j]
  #first find duplicated corners
  while(flag_merge):
    curve_corner_matching = (curve_corner_similarity > 0.5).astype(np.int32)
    conflict_curve_idx = np.where(curve_corner_matching.sum(axis=-1) > 2)[0]
    print(conflict_curve_idx)
    if(conflict_curve_idx.shape[0] == 0):
      break
    
    potential_duplicate_corners = np.where(curve_corner_matching[conflict_curve_idx[0]] > 0.5)[0]
    min_value = 1e10
    min_pair = None
    for i in range(potential_duplicate_corners.shape[0]):
      for j in range(i+1, potential_duplicate_corners.shape[0]):
        corner1 = potential_duplicate_corners[i]
        corner2 = potential_duplicate_corners[j]
        cur_diff = np.square(curve_corner_similarity[:, corner1] - curve_corner_similarity[:, corner2]).sum()
        if(cur_diff < min_value):
          min_value = cur_diff
          min_pair = (corner1, corner2)
          print(min_value, min_pair)
    data['corners']['prediction']['valid_prob'][min_pair[0]] = 0
    print(corners['valid_prob'][min_pair[0]])
    curve_corner_similarity[:, min_pair[0]] = 0
  return data


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = (np.mean(min_y_to_x) + np.mean(min_x_to_y))/2.0
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist

def NMS_patch(data):
  corners = data['corners']['prediction']
  curves = data['curves']['prediction']
  patches = data['patches']['prediction']
  valid_corners_idx = np.where(corners['valid_prob'] > th_valid)
  valid_curves_idx = np.where(curves['valid_prob'] > th_valid)
  valid_patches_idx = np.where(patches['valid_prob'] > th_valid)
  
  curve_corner_similarity = data['curve_corner_similarity']
  patch_curve_similarity = data['patch_curve_similarity']
  patch_corner_similarity = data['patch_corner_similarity']
  curve_corner_similarity_round = np.round(data['curve_corner_similarity'])
  patch_curve_similarity_round = np.round(data['patch_curve_similarity'])
  patch_corner_similarity_round = np.round(data['patch_corner_similarity'])
  #when deleting an item, the corresponding topo elements should be set as zeros
  
  patch_pts = patches['points']
  n_patch = len(valid_patches_idx[0])
  # alldists = []
  for i in range(n_patch - 1):
    if patches['valid_prob'][i] > th_valid:
      for j in range(i + 1, n_patch):
        tmp_chamfer_xy = chamfer_distance(patch_pts[i], patch_pts[j], direction='x_to_y')
        tmp_chamfer_yx = chamfer_distance(patch_pts[i], patch_pts[j], direction='y_to_x')
        if tmp_chamfer_xy < th_nms_dist_patch and tmp_chamfer_yx < th_nms_dist_patch:
          #distance close enough
          error_curve = np.abs(patch_curve_similarity_round[i] - patch_curve_similarity_round[j]).sum()
          error_corner = np.abs(patch_corner_similarity_round[i] - patch_corner_similarity_round[j]).sum()
          if error_curve == 0 and error_corner == 0:
            print('merge patch {} {}'.format(i,j))
            patches['valid_prob'][j] = 0
            patch_curve_similarity[j] = 0.0
            patch_corner_similarity[j] = 0.0


def NMS_curve(data):
  corners = data['corners']['prediction']
  curves = data['curves']['prediction']
  patches = data['patches']['prediction']
  valid_corners_idx = np.where(corners['valid_prob'] > th_valid)
  valid_curves_idx = np.where(curves['valid_prob'] > th_valid)
  valid_patches_idx = np.where(patches['valid_prob'] > th_valid)
  
  curve_corner_similarity = data['curve_corner_similarity']
  patch_curve_similarity = data['patch_curve_similarity']
  patch_corner_similarity = data['patch_corner_similarity']
  curve_corner_similarity_round = np.round(data['curve_corner_similarity'])
  patch_curve_similarity_round = np.round(data['patch_curve_similarity'])
  patch_corner_similarity_round = np.round(data['patch_corner_similarity'])
  #when deleting an item, the corresponding topo elements should be set as zeros
  
  curve_pts = curves['points']
  n_curve = len(valid_curves_idx[0])
  # alldists = []
  for i in range(n_curve - 1):
    # np.savetxt("{}_curve.xyz".format(i), curve_pts[i])
    if curves['valid_prob'][i] > th_valid:
      for j in range(i + 1, n_curve):
        tmp_chamfer = chamfer_distance(curve_pts[i], curve_pts[j])
        if tmp_chamfer < th_nms_dist:
          #distance close enough
          error_patch = np.abs(patch_curve_similarity_round[:,i] - patch_curve_similarity_round[:,j]).sum()
          error_corner = np.abs(curve_corner_similarity_round[i] - curve_corner_similarity_round[j]).sum()
          if error_patch == 0 and error_corner == 0:
            print('merge curve {} {}'.format(i,j))
            curves['valid_prob'][j] = 0
            patch_curve_similarity[:,j] = 0.0
            curve_corner_similarity[j] = 0.0

def NMS_corner(data):
  corners = data['corners']['prediction']
  curves = data['curves']['prediction']
  patches = data['patches']['prediction']
  valid_corners_idx = np.where(corners['valid_prob'] > th_valid)
  valid_curves_idx = np.where(curves['valid_prob'] > th_valid)
  valid_patches_idx = np.where(patches['valid_prob'] > th_valid)
  
  curve_corner_similarity = data['curve_corner_similarity']
  patch_curve_similarity = data['patch_curve_similarity']
  patch_corner_similarity = data['patch_corner_similarity']
  curve_corner_similarity_round = np.round(data['curve_corner_similarity'])
  patch_curve_similarity_round = np.round(data['patch_curve_similarity'])
  patch_corner_similarity_round = np.round(data['patch_corner_similarity'])
  
  corner_pts = corners['position']
  n_corner = len(valid_corners_idx[0])
  # alldists = []
  for i in range(n_corner - 1):
    if corners['valid_prob'][i] > th_valid:
      for j in range(i + 1, n_corner):
        tmp_chamfer = np.linalg.norm(corner_pts[i] - corner_pts[j])
        if tmp_chamfer < th_nms_dist:
          #distance close enough
          error_patch = np.abs(patch_corner_similarity_round[:,i] - patch_corner_similarity_round[:,j]).sum()
          error_curve = np.abs(curve_corner_similarity_round[:,i] - curve_corner_similarity_round[:,j]).sum()
          if error_patch == 0 and error_curve == 0:
            print('merge corner {} {}'.format(i,j))
            corners['valid_prob'][j] = 0
            patch_corner_similarity[:,j] = 0.0
            curve_corner_similarity[:,j] = 0.0

def NMS(data):
  #merge elements with similar geometry and same connection
  corners = data['corners']['prediction']
  curves = data['curves']['prediction']
  patches = data['patches']['prediction']
  
  curve_corner_similarity = data['curve_corner_similarity']
  patch_curve_similarity = data['patch_curve_similarity']
  patch_corner_similarity = data['patch_corner_similarity']

  valid_corners_idx = np.where(corners['valid_prob'] > th_valid)
  valid_curves_idx = np.where(curves['valid_prob'] > th_valid)
  valid_patches_idx = np.where(patches['valid_prob'] > th_valid)

  corners['valid_prob'] = corners['valid_prob'][valid_corners_idx]
  curves['valid_prob'] = curves['valid_prob'][valid_curves_idx]
  patches['valid_prob'] = patches['valid_prob'][valid_patches_idx]

  corners['position'] = corners['position'][valid_corners_idx]
  for k in curves:
    if k != 'valid_prob':
      curves[k] = curves[k][valid_curves_idx]
  
  for k in patches:
    if k != 'valid_prob':
      patches[k] = patches[k][valid_patches_idx]
  data['curve_corner_similarity'] = curve_corner_similarity[valid_curves_idx[0]][:, valid_corners_idx[0]]
  data['patch_curve_similarity'] = patch_curve_similarity[valid_patches_idx[0]][:, valid_curves_idx[0]]
  data['patch_corner_similarity'] = patch_corner_similarity[valid_patches_idx[0]][:, valid_corners_idx[0]]
  NMS_patch(data)
  NMS_curve(data)
  NMS_corner(data)
  

def closed_curve_distance(pts0, pts1):
  assert(pts0.shape == pts1.shape)
  pts0_flip = np.flip(pts0, 0)
  pts0_all = []
  for i in range(pts0.shape[0]):
    pts0_all.append(np.roll(pts0, i, axis = 0))
  
  for i in range(pts0.shape[0]):
    pts0_all.append(np.roll(pts0_flip, i, axis = 0))

  pts0_all = np.array(pts0_all)
  # print('pts0 all shape: ', pts0_all.shape)
  pts1_all = np.repeat(np.expand_dims(pts1, 0), 2 * pts0.shape[0], axis = 0)
  dist_all = np.linalg.norm(pts0_all - pts1_all, axis = -1).sum(-1) / pts0.shape[0]
  # print('dist all shape: ', dist_all.shape)
  return dist_all.min()

th_close_ratio = 0.1 #close curve's endpoints length/total length should be less than this threshold
th_merge = 0.05 #real close curves pairs with same topo and distance less than this threshold should be merged
th_open = 0.5 #threshold for deciding whether a curve is open or not

def check_feasibility(data):
  sample_id = data['sample_id']
  print("working on case {}".format(sample_id))
  
  #primitives
  corners = data['corners']['prediction']
  curves = data['curves']['prediction']
  patches = data['patches']['prediction']
  
  curve_corner_similarity = data['curve_corner_similarity']
  patch_curve_similarity = data['patch_curve_similarity']
  patch_corner_similarity = data['patch_corner_similarity'].astype(dtype)

  corners_valid_prob = corners['valid_prob'].astype(dtype)
  curves_valid_prob = curves['valid_prob'].astype(dtype)
  patches_valid_prob = patches['valid_prob'].astype(dtype)
  open_curve_prob = 1 - curves['closed_prob'].astype(dtype)
  #patch_corner_similarity = (np.clip(np.matmul(patch_curve_similarity, curve_corner_similarity) / 2.0, a_min=0, a_max=1) > 0.5).astype(np.float32)
  #print(patch_corner_similarity.min(), patch_corner_similarity.max())
  
  #data modified here
  assert(curve_corner_similarity.shape == patch_curve_similarity.shape)
  for i in range(curve_corner_similarity.shape[0]): #curve_id
    for j in range(curve_corner_similarity.shape[1]): #corner_id
      curve_corner_similarity[i][j] *= corners['valid_prob'][j] * curves['valid_prob'][i] * (1 - curves['closed_prob'][i])
      
  for i in range(patch_curve_similarity.shape[0]): #patch_id
    for j in range(patch_curve_similarity.shape[1]): #curve_id
      patch_curve_similarity[i][j] *= patches['valid_prob'][i] * curves['valid_prob'][j]
  
  for i in range(patch_corner_similarity.shape[0]):
    for j in range(patch_corner_similarity.shape[1]):
      patch_corner_similarity[i][j] *= patches['valid_prob'][i] * corners['valid_prob'][j]
  
  #first find duplicated corners
  flag_feas = True
  #rounding directly to check the feasibility
  #set A_ub before
  # np.savetxt('curve_corner_sim.txt', curve_corner_similarity, fmt='%.3f')
  # np.savetxt('curve_valid.txt', curves_valid_prob, fmt='%.3f')
  # np.savetxt('corner_valid.txt', curves_valid_prob, fmt='%.3f')


  #definition of C
  c = np.zeros(300+100+30000, dtype=dtype) #curve-corner patch-corner patch-curve
  c[:100] = corners_valid_prob
  c[100:200] = curves_valid_prob
  c[200:300] = patches_valid_prob
  # c[300:10300] = np.reshape(w_curvecorner_topo * curve_corner_similarity + w_curvecorner_geom * curve_corner_similarity_geom, [-1])
  c[300:10300] = np.reshape(curve_corner_similarity, [-1])
  c[20300:30300] = np.reshape(patch_curve_similarity, [-1])
  c[30300:] = open_curve_prob*curves_valid_prob
  c[10300:20300] = np.reshape(patch_corner_similarity, [-1])

  #round c to int
  c = np.round(c)
  A_ub = []
  A_eq = []
  b_eq = []
  cur_row = 0
  for i in range(100): #curve
    for j in range(100): #corner
      #EV(i, j) < E(i)
      oneline = np.zeros(c.shape[0], dtype=dtype)
      # A_ub[cur_row][100+i] = -1
      # A_ub[cur_row][300+100*i+j] = 1
      oneline[100 + i] = -1
      oneline[300 + 100*i + j] = 1
      A_ub.append(oneline)
      cur_row += 1
      
      #EV(i, j) < V(j)
      # A_ub[cur_row][j] = -1
      # A_ub[cur_row][300+100*i+j] = 1
      oneline = np.zeros(c.shape[0], dtype=dtype)
      oneline[j] = -1
      oneline[300 + 100*i + j] = 1
      A_ub.append(oneline)
      cur_row += 1
  
  for i in range(100): #patch
    for j in range(100): #curve
      #FE(i,j) < F(i)
      # A_ub[cur_row][200+i] = -1
      # A_ub[cur_row][20300+100*i+j] = 1

      oneline = np.zeros(c.shape[0], dtype=dtype)
      oneline[200 + i] = -1
      oneline[20300+100*i+j] = 1
      A_ub.append(oneline)
      cur_row += 1
      
      #FE(i,j) < E(j) # omit cause equation 1
      # A_ub[cur_row][100+j] = -1
      # A_ub[cur_row][20300+100*i+j] = 1
      # oneline = np.zeros(c.shape[0], dtype=np.float32)
      # oneline[100 + j] = -1
      # oneline[20300+100*i+j] = 1
      # A_ub.append(oneline)
      # cur_row += 1
  
  for i in range(100): #patch
    for j in range(100): #corner
      #FV(i,j) < F(i)
      # A_ub[cur_row][200+i] = -1
      # A_ub[cur_row][10300+100*i+j] = 1
      oneline = np.zeros(c.shape[0], dtype=dtype)
      oneline[200+i] = -1
      oneline[10300+100*i+j] = 1
      A_ub.append(oneline)
      cur_row += 1
      #FV(i,j) < V(j)
      # A_ub[cur_row][j] = -1
      # A_ub[cur_row][10300+100*i+j] = 1
      oneline = np.zeros(c.shape[0], dtype=dtype)
      oneline[j] = -1
      oneline[10300+100*i+j] = 1
      A_ub.append(oneline)
      cur_row += 1
  
  for i in range(100): #corner
    # A_ub[cur_row][i] = 1
    #V(i) < sigma EV
    oneline = np.zeros(c.shape[0], dtype=dtype)
    oneline[i] = 1
    for j in range(100): #curve
      # A_ub[cur_row][300+100*j+i] = -1
      oneline[300 + 100 * j + i] = -1
    A_ub.append(oneline)
    cur_row += 1
  
  for i in range(100): #patch
    # A_ub[cur_row][200+i] = 1
    oneline = np.zeros(c.shape[0], dtype=dtype)
    oneline[200 + i] = 1
    #F(i) < sigma FE
    for j in range(100): #curve
      # A_ub[cur_row][20300+100*i+j] = -1
      oneline[20300+100*i+j] = -1
    A_ub.append(oneline)
    cur_row += 1
  
  # for i in range(100): #patch
  #   # A_ub[cur_row][200+i] = 1
  #   oneline = np.zeros(c.shape[0], dtype=np.float32)
  #   oneline[200 + i] = 1
  #   #F(i) < sigma FV, strange
  #   for j in range(100): #corner
  #     # A_ub[cur_row][10300+100*i+j] = -1
  #     oneline[10300+100*i+j] = -1
  #   A_ub.append(oneline)
  #   cur_row += 1
  
  for i in range(100): #corner
    # A_ub[cur_row][i] = 1
    oneline = np.zeros(c.shape[0], dtype=dtype)
    oneline[i] = 1
    #V(i) < sigma FV
    for j in range(100): #patch
      # A_ub[cur_row][10300+100*j+i] = -1
      oneline[10300+100*j+i] = -1
    A_ub.append(oneline)
    cur_row += 1
  #curve open_prob < valid_prob
  for i in range(100):
    #O < E
    # A_ub[cur_row][100+i] = -1
    # A_ub[cur_row][30300+i] = 1
    oneline = np.zeros(c.shape[0], dtype=dtype)
    oneline[100+i] = -1
    oneline[30300+i] = 1
    A_ub.append(oneline)
    cur_row += 1
  
  A_ub = np.array(A_ub)
  b_ub = np.zeros(A_ub.shape[0], dtype=dtype)
  print("A ub shape: ", A_ub.shape)
  assert(cur_row == A_ub.shape[0])
  
  #set part of A_eq
  cur_row = 0
  #curve patch constraints (every curve has 2 patch)
  for i in range(100): #curve
    # A_eq[cur_row][100+i] = 2
    oneline = np.zeros(c.shape[0], dtype=dtype)
    oneline[100 + i] = 2
    for j in range(100): #patch
      # A_eq[cur_row][20300+100*j+i] = -1
      oneline[20300+100*j+i] = -1
    A_eq.append(oneline)
    b_eq.append(0.0)
    cur_row += 1
  
  #curve - 2 corner constraints
  for i in range(100): #curve
    # A_eq[cur_row][30300+i] = 2 #no validness
    oneline = np.zeros(c.shape[0], dtype=dtype)
    oneline[30300+i] = 2 #no validness
    for j in range(100): #corner
      # A_eq[cur_row][300+ 100*i +j] = -1
      oneline[300+ 100*i +j] = -1
    A_eq.append(oneline)
    b_eq.append(0.0)
    cur_row += 1


  #bnb
  opt_x = np.zeros(c.shape[0])
  obj_ub = 10.0 #upper bound
  conf_list = []
  A_eq_base = np.array(A_eq)
  b_eq_base = np.array(b_eq)

  #outer iter
  x_init = c
  outer_iter_count = 0
  #try twice: first free all variables, then fix patch_curve_similarity
  # while outer_iter_count < 2:
  if True:
    #reset pc, cc, pc similarity
    curve_corner_similarity_tmp = np.round(curve_corner_similarity)
    
    A_eq_out = copy.deepcopy(A_eq)
    b_eq_out = copy.deepcopy(b_eq)
    #inner iter
    if True:
      # print("iter", iter)
      # assert(fixed_variable_set == 'CC' or fixed_variable_set == 'PC')
      #constraints
      # A_ub[:] = 0
      # b_ub[:] = 0
      #equalities
      # A_eq[:] = 0
      # b_eq[:] = 0
      #FE*EC = 2 *FC
      A_eq_copy = copy.deepcopy(A_eq_out)
      b_eq_copy = copy.deepcopy(b_eq_out)
      for i in range(100): #patch
        for j in range(100): #corner
          # A_eq[cur_row][10300 + 100*i + j] = -2
          oneline = np.zeros(c.shape[0], dtype=dtype)
          oneline[10300 + 100*i + j] = -2
          for k in range(100): #curve
            #2 * patch_corner[i][j] = \sum patch_curve[i][k] * curve_corner[k][j]
            oneline[20300+100*i+k] = curve_corner_similarity_tmp[k][j]
          A_eq_copy.append(oneline)
          b_eq_copy.append(0.0)
          cur_row += 1
      
      A_eq_copy = np.array(A_eq_copy)
      b_eq_copy = np.array(b_eq_copy)
      # assert(cur_row == A_eq.shape[0])
      b_eq = np.zeros(A_eq_copy.shape[0], dtype=dtype)

      #unequal check:
      if (b_ub - np.matmul(A_ub, c) >= 0.0).min() == False:
        flag_feas = False
        print("unequality status: ", flag_feas)
        print("max sum: {} max id: {}".format(np.matmul(A_ub, c).max(), np.argmax(np.matmul(A_ub, c))) )
     

      if (np.abs(b_eq - np.matmul(A_eq_copy, c)) < 0.1).min() == False:
        flag_feas = False
        print("unsatisfied equals")
        print("max sum: {} max id: {}".format(np.matmul(A_eq_copy, c).max(), np.argmax(np.matmul(A_eq_copy, c))))
        print("min sum: {} min id: {}".format(np.matmul(A_eq_copy, c).min(), np.argmin(np.matmul(A_eq_copy, c))))


      # res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(0,1), method='highs-ipm', options={'disp':True, 'dual_feasibility_tolerance':1e-16, 'primal_feasibility_tolerance':1e-16})
      
      #for model 06
      # res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(0,1), method='highs-ipm', options={'disp':True, 'dual_feasibility_tolerance':1e-7, 'primal_feasibility_tolerance':1.25e-8})
      # xbd = np.zeros([c.shape[0],2])
      # xbd[:,1] = 1.0
      # xbdtype = 3 * np.ones(c.shape[0], dtype = np.int32)
      # if iter < float_iter:
      #   (flag_opt_fea, x) = mosek_linprog(c, A_ub, b_ub, A_eq_copy, b_eq_copy, xbd, xbdtype, flag_max = False, flag_int = False)
      # else:
      #   #int solver with solution guess
      #   (flag_opt_fea, x) = mosek_linprog(c, A_ub, b_ub, A_eq_copy, b_eq_copy, xbd, xbdtype, flag_max = False, flag_int = True, flag_initial_guess = True, initial_guess = x_init)

  return flag_feas

def process_sample(data):
  sample_id = data['sample_id']
  print("working on case {}".format(sample_id))
  
  #primitives
  corners = data['corners']['prediction']
  curves = data['curves']['prediction']
  patches = data['patches']['prediction']
  
  valid_corners = []
  valid_curves = []
  for i in range(100):
    if(corners['valid_prob'][i] > 0.5):
      valid_corners.append((i, corners['valid_prob'][i]))
  
  for i in range(100):
    if(curves['valid_prob'][i] > 0.5):
      valid_curves.append((i, curves['valid_prob'][i]))
  
  print("{} valid corners".format(len(valid_corners)))
  print(valid_corners)
  min_distance = 1e10
  for i in valid_corners:
    for j in valid_corners:
      if(i[0] >= j[0]): continue
      distance = np.square(corners['position'][i[0]] - corners['position'][j[0]]).sum()
      if(distance < min_distance):
        min_distance = distance
        min_distance_pair = (i[0], j[0])
  
  print(min_distance, min_distance_pair)
  
  print("{} valid curves".format(len(valid_curves)))
  # print(valid_curves)
  input()
  
  curve_corner_similarity = data['curve_corner_similarity']
  patch_curve_similarity = data['patch_curve_similarity']
  
  valid_curve_corner_similarity = (curve_corner_similarity[[item[0] for item in valid_curves]])[:, [item[0] for item in valid_corners]]
  print(valid_curve_corner_similarity.shape)
  #print(valid_curve_corner_similarity)
  print(valid_curve_corner_similarity[:,0])
  print(valid_curve_corner_similarity[:,11])
  input()
  
  entry_list = []
  
  assert(curve_corner_similarity.shape == patch_curve_similarity.shape)
  for i in range(curve_corner_similarity.shape[0]): #curve_id
    for j in range(curve_corner_similarity.shape[1]): #corner_id
      prob = curve_corner_similarity[i][j] * corners['valid_prob'][j] * curves['valid_prob'][i] * (1 - curves['closed_prob'][i])
      entry_list.append((prob, 'CC', {'curve': i, 'corner': j}))
      
  for i in range(patch_curve_similarity.shape[0]): #patch_id
    for j in range(patch_curve_similarity.shape[1]): #curve_id
      prob = patch_curve_similarity[i][j] * patches['valid_prob'][i] * curves['valid_prob'][j]
      entry_list.append((prob, 'PC', {'patch': i, 'curve' :j}))
  
  #sort the entry_list
  sorted_entry_list = sorted(entry_list, key = lambda s: s[0], reverse=True)
  assert(len(sorted_entry_list) == 20000)
  
  for i in range(len(sorted_entry_list)):
    sorted_entry_list[i] += (i,)
  
  print(sorted_entry_list[0])
  print(sorted_entry_list[60])
  print(sorted_entry_list[70])
  print(sorted_entry_list[75])
  print(sorted_entry_list[80])
  print(sorted_entry_list[100])
  print(sorted_entry_list[1000])
  input()
  
  def primitives_in_stack(stack):
    corner_list = []
    curve_list = []
    patch_list = []    
    curve_corner_corres = {}
    patch_curve_corres = {}
    
    for entry in stack:
      if(entry[2]['curve'] not in curve_list):
        curve_list.append(entry[2]['curve'])
      if(entry[1] == 'CC'):
        if(entry[2]['corner'] not in corner_list):
          corner_list.append(entry[2]['corner'])
        if(entry[2]['curve'] not in curve_corner_corres):
          curve_corner_corres[entry[2]['curve']] = []
        curve_corner_corres[entry[2]['curve']].append(entry[2]['corner'])
      else:
        assert(entry[1] == 'PC')
        if(entry[2]['patch'] not in patch_list):
          patch_list.append(entry[2]['patch'])
        if(entry[2]['patch'] not in patch_curve_corres):
          patch_curve_corres[entry[2]['patch']] = []
        patch_curve_corres[entry[2]['patch']].append(entry[2]['curve'])
    return corner_list, curve_list, patch_list, curve_corner_corres, patch_curve_corres
  
  #Search
  stack = []
  corners_in_curve = np.zeros([100], dtype=np.int32)
  patches_with_curve = np.zeros([100], dtype=np.int32)
  i = 0
  while(i < 20000):
    entry = sorted_entry_list[i]
    consistent = True
    curve_id = entry[2]['curve']
    if(entry[1] == 'CC'):
      if(corners_in_curve[curve_id] >= 2):
        consistent = False
    else:
      assert(entry[1] == 'PC')
      if(patches_with_curve[curve_id] >= 2):
        consistent = False
    
    if(consistent):
      stack.append(entry)
      if(entry[1] == 'CC'):
        corners_in_curve[curve_id] += 1
      else:
        patches_with_curve[curve_id] += 1
      
      #check if current state is complete
      complete = True
      corner_list, curve_list, patch_list, curve_corner_corres, patch_curve_corres = primitives_in_stack(stack)
      print("{} corners {} curves and {} patches".format(len(corner_list), len(curve_list), len(patch_list)))
      #If every curve has 2 corners
      for entry in stack:          
        if(corners_in_curve[entry[2]['curve']] != 2):
          if(corners_in_curve[entry[2]['curve']] != 0):
            complete =False
            break
          else:
            if(curves['closed_prob'][entry[2]['curve']] < 1e-2):
              complete = False
            elif(curves['closed_prob'][entry[2]['curve']] < 0.1):
              print('detect closed curve {} with prob {}'.format(entry[2]['curve'], curves['closed_prob'][entry[2]['curve']]))
      if(complete):
        #input()
        #If curves occurred in 2 patches(necessary? What if shape is open, maybe <=2(already satisfied in search))
        for entry in stack:          
          if(patches_with_curve[entry[2]['curve']] != 2):
            complete =False
            break
        if(complete):
          #If curves of patches is closed (patch boundary)
          corner_list, curve_list, patch_list, curve_corner_corres, patch_curve_corres = primitives_in_stack(stack)
          print("{} corners {} curves and {} patches".format(len(corner_list), len(curve_list), len(patch_list)))
          for patch in patch_list:
            corners = {}
            for curve in patch_curve_corres[patch]:
              if(curve in curve_corner_corres):
                for corner in curve_corner_corres[curve]:
                  if(corner in corners):
                    corners[corner] += 1
                  else:
                    corners[corner] = 1
            for corner in corners:
              if(corners[corner] % 2 == 1):
                complete =False
            
          if(complete):
            print("Complete")
            print("Stack Size = {}".format(len(stack)))
            print("{} corners {} curves and {} patches".format(len(corner_list), len(curve_list), len(patch_list)))
            break      
      
    while(i+1 == 20000 or sorted_entry_list[i+1][0] < 0.5):
      #back trace
      if(len(stack) == 0):
        print("end of search: no feasible solution")
        return
      entry = stack.pop()
      i = entry[3]
      print("backtrace to {} stack size = {}".format(i+1, len(stack)))
      curve_id = entry[2]['curve']
      if(entry[1] == 'CC'):
        corners_in_curve[curve_id] -= 1
      else:
        patches_with_curve[curve_id] -= 1
      
    i += 1

def export_visualization_file(output_filename, data):
  def id_to_curve_type(id):
    #Circle, BSpline, Line, Ellipse
    if(id == 0):
      return 'Circle'
    if(id == 1):
      return 'BSpline'
    if(id == 2):
      return 'Line'
    assert(id == 3)
    return 'Ellipse'
  
  def id_to_patch_type(str):
    #Cylinder, Torus, BSpline, Plane, Cone, Sphere
    if(str == 0):
      return 'Cylinder'
    if(str == 1):
      return 'Torus'
    if(str == 2):
      return 'BSpline'
    if(str == 3):
      return 'Plane'
    if(str == 4):
      return 'Cone'
    assert(str == 5)
    return 'Sphere'
  
  sample_id = data['sample_id']
  print("working on case {}".format(sample_id))
  
  #primitives
  corners = data['corners']['prediction']
  curves = data['curves']['prediction']
  patches = data['patches']['prediction']
  
  if 'curve_corner_similarity' in data:
    curve_corner_similarity = data['curve_corner_similarity']
    patch_curve_similarity = data['patch_curve_similarity']
    patch_corner_similarity = data['patch_corner_similarity']
  else:
    #compute similarity according to geometry and store them in data
    curve_corner_similarity = get_curve_corner_similarity_geom(data)
    patch_curve_similarity = get_patch_curve_similarity_geom(data)
    patch_corner_similarity = get_patch_corner_similarity_geom(data)
    data['curve_corner_similarity'] = curve_corner_similarity
    data['patch_curve_similarity'] = patch_curve_similarity
    data['patch_corner_similarity'] = patch_corner_similarity
  flag_with_normal = False
  if 'normals' in patches:
    flag_with_normal = True
  
  flag_save_patch_corner = True
  flag_patch_close = False
  if 'closed_prob' in patches:
    flag_patch_close = True
  
  with open(output_filename, "w") as wf:
    valid_corners_idx = np.where(corners['valid_prob'] > th_valid)
    valid_curves_idx = np.where(curves['valid_prob'] > th_valid)
    valid_patches_idx = np.where(patches['valid_prob'] > th_valid)
    valid_patch_points = patches['points'][valid_patches_idx]
    if len(valid_patch_points) > 0:
      grid_size = len(valid_patch_points[0])
      if grid_size == 400:
        wf.write("-1\n") #flag for 20x20

    wf.write("{} {} {}\n".format(valid_corners_idx[0].shape[0], valid_curves_idx[0].shape[0], valid_patches_idx[0].shape[0]))
    
    valid_corners = corners['position'][valid_corners_idx]  
    assert(valid_corners.shape[0] == valid_corners_idx[0].shape[0])
    for corner in valid_corners:
      wf.write("{:.6f} {:.6f} {:.6f}\n".format(corner[0], corner[1], corner[2]))
    
    valid_curve_type = np.argmax(curves['type_prob'], axis=-1)[valid_curves_idx]
    assert(valid_curve_type.shape[0] == valid_curves_idx[0].shape[0])
    
    valid_curve_points = curves['points'][valid_curves_idx]
    valid_curve_closed_info = curves['closed_prob'][valid_curves_idx]
    print(valid_curve_points.shape)
    assert(valid_curve_points.shape[0] == valid_curve_closed_info.shape[0])
    for i in range(valid_curve_type.shape[0]):
      # wf.write("{} {} ".format(id_to_curve_type(valid_curve_type[i]), int(valid_curve_closed_info[i] > 0.5)))
      wf.write("{} {} ".format(id_to_curve_type(valid_curve_type[i]), valid_curve_closed_info[i]))
      for curve_point in valid_curve_points[i]:
        wf.write("{:.6f} {:.6f} {:.6f} ".format(curve_point[0], curve_point[1], curve_point[2]))
      wf.write("\n")
    
    valid_patch_type = np.argmax(patches['type_prob'], axis=-1)[valid_patches_idx]
    print(valid_patch_points.shape)
    assert(valid_patch_type.shape[0] == valid_patches_idx[0].shape[0])
    for i in range(valid_patch_type.shape[0]):
      wf.write("{} ".format(id_to_patch_type(valid_patch_type[i])))
      for patch_point in valid_patch_points[i]:
        wf.write("{:.6f} {:.6f} {:.6f} ".format(patch_point[0], patch_point[1], patch_point[2]))
      wf.write("\n")
    
    print('curve corner sim shape: ',curve_corner_similarity.shape)
    valid_curve_corner_similarity = np.squeeze(((curve_corner_similarity[valid_curves_idx][:, valid_corners_idx]) > 0.5).astype(np.int32), axis=1)
    print(valid_curve_corner_similarity.shape)
    for i in range(valid_curve_corner_similarity.shape[0]):
      for j in range(valid_curve_corner_similarity.shape[1]):
        wf.write("{} ".format(valid_curve_corner_similarity[i][j]))
      wf.write("\n")
    
    valid_patch_curve_similarity = np.squeeze(((patch_curve_similarity[valid_patches_idx][:, valid_curves_idx]) > 0.5).astype(np.int32), axis=1)
    for i in range(valid_patch_curve_similarity.shape[0]):
      for j in range(valid_patch_curve_similarity.shape[1]):
        wf.write("{} ".format(valid_patch_curve_similarity[i][j]))
      wf.write("\n")
    
    if flag_with_normal:
      wf.write('{}\n'.format(int(flag_with_normal)))
      valid_patch_normals = patches['normals'][valid_patches_idx]
      # print(valid_patch_points.shape)
      # assert(valid_patch_type.shape[0] == valid_patches_idx[0].shape[0])
      for i in range(valid_patch_type.shape[0]):
        wf.write("{} ".format(id_to_patch_type(valid_patch_type[i])))
        for patch_normal in valid_patch_normals[i]:
          wf.write("{:.6f} {:.6f} {:.6f} ".format(patch_normal[0], patch_normal[1], patch_normal[2]))
        wf.write("\n")
    if flag_patch_close:
      valid_patch_close = patches['closed_prob'][valid_patches_idx]
      for i in range(valid_patch_type.shape[0]):
        wf.write("{:.6f}\n".format(valid_patch_close[i]))
    
    if flag_save_patch_corner:
      valid_patch_corner_similarity = np.squeeze(((patch_corner_similarity[valid_patches_idx][:, valid_corners_idx]) > 0.5).astype(np.int32), axis=1)
      for i in range(valid_patch_corner_similarity.shape[0]):
        for j in range(valid_patch_corner_similarity.shape[1]):
          wf.write("{} ".format(valid_patch_corner_similarity[i][j]))
        wf.write("\n")

def export_visualization_file_gt(output_filename, data):
  def id_to_curve_type(id):
    #Circle, BSpline, Line, Ellipse
    if(id == 0):
      return 'Circle'
    if(id == 1):
      return 'BSpline'
    if(id == 2):
      return 'Line'
    assert(id == 3)
    return 'Ellipse'
  
  def id_to_patch_type(str):
    #Cylinder, Torus, BSpline, Plane, Cone, Sphere
    if(str == 0):
      return 'Cylinder'
    if(str == 1):
      return 'Torus'
    if(str == 2):
      return 'BSpline'
    if(str == 3):
      return 'Plane'
    if(str == 4):
      return 'Cone'
    assert(str == 5)
    return 'Sphere'
  
  sample_id = data['sample_id']
  print("working on case {}".format(sample_id))
  
  #primitives
  corners = data['corners']['gt']
  curves = data['curves']['gt']
  patches = data['patches']['gt']
  
  curve_corner_similarity = data['curve_corner_correspondence_gt']
  patch_curve_similarity = data['patch_curve_correspondence_gt']
  flag_with_normal = False
  # if 'normals' in patches:
  #   flag_with_normal = True
  
  flag_patch_close = False
  # if 'closed_prob' in patches:
  #   flag_patch_close = True
  
  with open(output_filename, "w") as wf:
    patch_pts = patches['points']
    patch_type = patches['type']
    curve_pts = curves['points']
    curve_type = curves['type']
    curve_close = curves['is_closed']
    corner_pts = corners
    if len(patch_pts) > 0:
      grid_size = len(patch_pts[0])
      if grid_size == 400:
        wf.write("-1\n") #flag for 20x20
    wf.write("{} {} {}\n".format(corner_pts.shape[0], curve_pts.shape[0], len(patch_pts)))
    
    for i in range(len(corner_pts)):
      corner = corner_pts[i]
      wf.write("{:.6f} {:.6f} {:.6f}\n".format(corner[0], corner[1], corner[2]))
    
    for i in range(len(curve_pts)):
      # wf.write("{} {} ".format(id_to_curve_type(valid_curve_type[i]), int(valid_curve_closed_info[i] > 0.5)))
      wf.write("{} {} ".format(id_to_curve_type(curve_type[i]), curve_close[i]))
      for j in range(len(curve_pts[i])):
        curve_point = curve_pts[i][j]
        wf.write("{:.6f} {:.6f} {:.6f} ".format(curve_point[0], curve_point[1], curve_point[2]))
      wf.write("\n")
    

    for i in range(len(patch_pts)):
      wf.write("{} ".format(id_to_patch_type(patch_type[i])))
      for j in range(len(patch_pts[i])):
        patch_point = patch_pts[i][j]
        wf.write("{:.6f} {:.6f} {:.6f} ".format(patch_point[0], patch_point[1], patch_point[2]))
      wf.write("\n")

    valid_patch_curve_similarity = patch_curve_similarity
    valid_curve_corner_similarity = curve_corner_similarity
    for i in range(valid_curve_corner_similarity.shape[0]):
      for j in range(valid_curve_corner_similarity.shape[1]):
        wf.write("{} ".format(valid_curve_corner_similarity[i][j]))
      wf.write("\n")
    for i in range(valid_patch_curve_similarity.shape[0]):
      for j in range(valid_patch_curve_similarity.shape[1]):
        wf.write("{} ".format(valid_patch_curve_similarity[i][j]))
      wf.write("\n")
  return

  with open(output_filename, "w") as wf:
    valid_corners_idx = np.where(corners['valid_prob'] > th_valid)
    valid_curves_idx = np.where(curves['valid_prob'] > th_valid)
    valid_patches_idx = np.where(patches['valid_prob'] > th_valid)
    valid_patch_points = patches['points'][valid_patches_idx]
    if len(valid_patch_points) > 0:
      grid_size = len(valid_patch_points[0])
      if grid_size == 400:
        wf.write("-1\n") #flag for 20x20

    wf.write("{} {} {}\n".format(valid_corners_idx[0].shape[0], valid_curves_idx[0].shape[0], valid_patches_idx[0].shape[0]))
    
    valid_corners = corners['position'][valid_corners_idx]  
    assert(valid_corners.shape[0] == valid_corners_idx[0].shape[0])
    for corner in valid_corners:
      wf.write("{:.6f} {:.6f} {:.6f}\n".format(corner[0], corner[1], corner[2]))
    
    valid_curve_type = np.argmax(curves['type_prob'], axis=-1)[valid_curves_idx]
    assert(valid_curve_type.shape[0] == valid_curves_idx[0].shape[0])
    
    valid_curve_points = curves['points'][valid_curves_idx]
    valid_curve_closed_info = curves['closed_prob'][valid_curves_idx]
    print(valid_curve_points.shape)
    assert(valid_curve_points.shape[0] == valid_curve_closed_info.shape[0])
    for i in range(valid_curve_type.shape[0]):
      # wf.write("{} {} ".format(id_to_curve_type(valid_curve_type[i]), int(valid_curve_closed_info[i] > 0.5)))
      wf.write("{} {} ".format(id_to_curve_type(valid_curve_type[i]), valid_curve_closed_info[i]))
      for curve_point in valid_curve_points[i]:
        wf.write("{:.6f} {:.6f} {:.6f} ".format(curve_point[0], curve_point[1], curve_point[2]))
      wf.write("\n")
    
    valid_patch_type = np.argmax(patches['type_prob'], axis=-1)[valid_patches_idx]
    print(valid_patch_points.shape)
    assert(valid_patch_type.shape[0] == valid_patches_idx[0].shape[0])
    for i in range(valid_patch_type.shape[0]):
      wf.write("{} ".format(id_to_patch_type(valid_patch_type[i])))
      for patch_point in valid_patch_points[i]:
        wf.write("{:.6f} {:.6f} {:.6f} ".format(patch_point[0], patch_point[1], patch_point[2]))
      wf.write("\n")
    
    print(curve_corner_similarity.shape)
    valid_curve_corner_similarity = np.squeeze(((curve_corner_similarity[valid_curves_idx][:, valid_corners_idx]) > 0.5).astype(np.int), axis=1)
    print(valid_curve_corner_similarity.shape)
    for i in range(valid_curve_corner_similarity.shape[0]):
      for j in range(valid_curve_corner_similarity.shape[1]):
        wf.write("{} ".format(valid_curve_corner_similarity[i][j]))
      wf.write("\n")
    
    valid_patch_curve_similarity = np.squeeze(((patch_curve_similarity[valid_patches_idx][:, valid_curves_idx]) > 0.5).astype(np.int), axis=1)
    for i in range(valid_patch_curve_similarity.shape[0]):
      for j in range(valid_patch_curve_similarity.shape[1]):
        wf.write("{} ".format(valid_patch_curve_similarity[i][j]))
      wf.write("\n")
    
    if flag_with_normal:
      wf.write('{}\n'.format(int(flag_with_normal)))
      valid_patch_normals = patches['normals'][valid_patches_idx]
      # print(valid_patch_points.shape)
      # assert(valid_patch_type.shape[0] == valid_patches_idx[0].shape[0])
      for i in range(valid_patch_type.shape[0]):
        wf.write("{} ".format(id_to_patch_type(valid_patch_type[i])))
        for patch_normal in valid_patch_normals[i]:
          wf.write("{:.6f} {:.6f} {:.6f} ".format(patch_normal[0], patch_normal[1], patch_normal[2]))
        wf.write("\n")
    if flag_patch_close:
      valid_patch_close = patches['closed_prob'][valid_patches_idx]
      for i in range(valid_patch_type.shape[0]):
        wf.write("{:.6f}\n".format(valid_patch_close[i]))

def load_complex_file(fn):
  #return a dictionary
  data = {'corners':{'prediction':{}}, 'curves':{'prediction':{}}, 'patches':{'prediction':{}}}

  n_curve_pts = 34
  n_patch_pts = 100
  f = open(fn, 'r')
  lines = f.readlines()
  f.close()
  cur_line = 0
  cur_split = lines[cur_line].split(' ')
  cur_line += 1
  nv = int(cur_split[0])
  ne = int(cur_split[1])
  nf = int(cur_split[2])
  corner_pos = np.zeros([nv,3])
  for i in range(nv):
    cur_split = lines[cur_line].split(' ')
    cur_line += 1
    for j in range(3):
      corner_pos[i,j] = float(cur_split[j])
  
  curve_type = []
  curve_close_prob = []
  curve_pts = np.zeros([ne, n_curve_pts, 3])
  for i in range(ne):
    cur_split = lines[cur_line].split(' ')
    cur_line += 1
    curve_type.append(cur_split[0])
    curve_close_prob.append(float(cur_split[1]))
    for j in range(n_curve_pts):
      for k in range(3):
        curve_pts[i,j,k] = float(cur_split[2 + 3 * j + k])
  
  patch_type = []
  patch_pts = np.zeros([nf, n_patch_pts, 3])
  for i in range(nf):
    cur_split = lines[cur_line].split(' ')
    cur_line += 1
    patch_type.append(cur_split[0])
    for j in range(n_patch_pts):
      for k in range(3):
        patch_pts[i,j,k] = float(cur_split[1 + 3 * j + k])
  
  curve_corner_similarity = np.zeros([ne, nv])
  for i in range(ne):
    cur_split = lines[cur_line].split(' ')
    cur_line += 1
    for j in range(nv):
      curve_corner_similarity[i,j] = float(cur_split[j])
  
  patch_curve_similarity = np.zeros([nf, ne])

  for i in range(nf):
    cur_split = lines[cur_line].split(' ')
    cur_line += 1
    for j in range(ne):
      patch_curve_similarity[i,j] = float(cur_split[j])
  
  # print('patch type: ', patch_type)
  # print('patch curve sim: ', patch_curve_similarity)
  corner_pred = {'valid_prob':np.ones(nv), 'position':corner_pos}
  curve_pred = {'valid_prob':np.ones(ne), 'closed_prob': np.array(curve_close_prob), 'points':curve_pts, 'type_prob': np.zeros([ne, 4])}
  patch_pred = {'valid_prob': np.ones(nf), 'points': patch_pts, 'type_prob': np.zeros([nf, 6])}
  curve_typemap = {'Circle':0, 'BSpline':1, 'Line':2, 'Ellipse':3}
  patch_typemap = {'Cylinder':0, 'Torus': 1, 'BSpline': 2, 'Plane':3, 'Cone': 4, 'Sphere': 5}
  for i in range(ne):
    curve_pred['type_prob'][i,curve_typemap[curve_type[i]]] = 1
  for i in range(nf):
    patch_pred['type_prob'][i,patch_typemap[patch_type[i]]] = 1
    
  data['corners']['prediction'] = corner_pred
  data['curves']['prediction'] = curve_pred
  data['patches']['prediction'] = patch_pred
  data['curve_corner_similarity'] = curve_corner_similarity
  data['patch_curve_similarity'] = patch_curve_similarity
  data['sample_id'] = fn
  return data

def load_json_file(fn):
  #return a dictionary
  data = {'corners':{'prediction':{}}, 'curves':{'prediction':{}}, 'patches':{'prediction':{}}}

  f = open(fn)
  info = json.load(f)
  f.close()

  n_curve_pts = 34
  n_patch_pts = 400

  nv = 0
  if info['corners'] != None:
    nv = len(info["corners"])
  ne = 0
  if info['curves'] != None:
    ne = len(info['curves'])

  nf = 0
  if info['patches'] != None:
    nf = len(info['patches'])  

  corner_pos = np.zeros([nv,3])
  for i in range(nv):
    for j in range(3):
      corner_pos[i,j] = info['corners'][i]['pts'][j]
  
  curve_type = []
  curve_close_prob = []
  curve_pts = np.zeros([ne, n_curve_pts, 3])
  for i in range(ne):
    curve_type.append(info['curves'][i]['type'])
    if info['curves'][i]['closed']:
      curve_close_prob.append(1.0)
    else:
      curve_close_prob.append(0.0)
    for j in range(n_curve_pts):
      for k in range(3):
        curve_pts[i,j,k] = info['curves'][i]['pts'][3 * j + k]
  
  patch_type = []
  patch_pts = np.zeros([nf, n_patch_pts, 3])
  patch_close = np.zeros(nf)
  patches = info['patches']
  for i in range(nf):
    patch_type.append(patches[i]['type'])
    if patches[i]['u_closed']:
      patch_close[i] = 1.0
    else:
      patch_close[i] = 0.0
    for j in range(n_patch_pts):
      for k in range(3):
        patch_pts[i,j,k] = patches[i]['grid'][3 * j + k]
  
  curve_corner_similarity = np.zeros([ne, nv], dtype=dtype)
  if info['curve2corner'] != None:
    curve_corner_similarity = np.array(info['curve2corner'])
  
  # patch_curve_similarity = np.zeros([nf, ne])
  patch_curve_similarity = np.zeros([nf, ne], dtype = dtype)
  if info['patch2curve'] != None:
    patch_curve_similarity = np.array(info['patch2curve'])

  patch_corner_similarity = np.zeros([nf, nv], dtype = dtype)
  if info['patch2corner'] != None:
    patch_corner_similarity = np.array(info['patch2corner'])

  corner_pred = {'valid_prob':np.ones(nv), 'position':corner_pos}
  curve_pred = {'valid_prob':np.ones(ne), 'closed_prob': np.array(curve_close_prob), 'points':curve_pts, 'type_prob': np.zeros([ne, 4])}
  patch_pred = {'valid_prob': np.ones(nf), 'points': patch_pts, 'type_prob': np.zeros([nf, 6]), 'closed_prob': patch_close}
  curve_typemap = {'Circle':0, 'BSpline':1, 'Line':2, 'Ellipse':3}
  patch_typemap = {'Cylinder':0, 'Torus': 1, 'BSpline': 2, 'Plane':3, 'Cone': 4, 'Sphere': 5}
  for i in range(ne):
    curve_pred['type_prob'][i,curve_typemap[curve_type[i]]] = 1
  for i in range(nf):
    patch_pred['type_prob'][i,patch_typemap[patch_type[i]]] = 1
    
  data['corners']['prediction'] = corner_pred
  data['curves']['prediction'] = curve_pred
  data['patches']['prediction'] = patch_pred
  data['curve_corner_similarity'] = curve_corner_similarity
  data['patch_curve_similarity'] = patch_curve_similarity
  data['patch_corner_similarity'] = patch_corner_similarity
  data['sample_id'] = fn
  return data

th_dist = 0.1
def check_geom_topo_cons(data):
  # res = 0
  corner_pts = data['corners']['prediction']['position']
  curve_pts = data['curves']['prediction']['points']
  patch_pts = data['patches']['prediction']['points']
  curve_close = data['curves']['prediction']['closed_prob']
  curve_corner_similarity = data['curve_corner_similarity']
  patch_curve_similarity = data['patch_curve_similarity']
  patch_corner_similarity = np.matmul(patch_curve_similarity, curve_corner_similarity) /2.0
  
  nv = corner_pts.shape[0]
  ne = curve_pts.shape[0]
  nf = patch_pts.shape[0]
  #curve corner diff
  for i in range(ne):
    if curve_close[i] < 0.5:
      curve_e1 = curve_pts[i][0]
      curve_e2 = curve_pts[i][-1]
      for j in range(nv):
        if curve_corner_similarity[i, j] > 0.5:
          cur_corner = corner_pts[j]
          if not (np.linalg.norm(curve_e1 - cur_corner) < th_dist or np.linalg.norm(curve_e2 - cur_corner) < th_dist):
            print('curve id: {} corner id: {}'.format(i,j))
            return 1
  
  #patch curve distance
  for i in range(nf):
    cur_patch_pts = patch_pts[i]
    for j in range(ne):
      if patch_curve_similarity[i, j] > 0.5:
        cur_curve_pts = curve_pts[j]
        pts_diff = np.expand_dims(cur_curve_pts, 1) - cur_patch_pts
        # print('pts diff shape: ', pts_diff.shape)
        pts_dist = np.linalg.norm(pts_diff, axis = -1)
        cur_dist = np.mean(pts_dist.min(-1))
        if cur_dist > th_dist:
          print('patch id: {} curve id: {} dist: {}'.format(i,j, cur_dist))
          return 2
          # b = 1
  
  for i in range(nf):
    cur_patch_pts = patch_pts[i]
    for j in range(nv):
      if patch_corner_similarity[i,j] > 0.5:
        cur_corner = corner_pts[j]
        pts_diff = cur_patch_pts - cur_corner
        pts_dist = np.linalg.norm(pts_diff, axis = -1)
        cur_dist = pts_dist.min()
        if cur_dist > th_dist:
          print('patch id: {} corner id: {} dist: {}'.format(i,j, cur_dist))
          return 3

  return 0



def process_one_file(filename):
  fn_prefix = filename.replace('_prediction.pkl','')
  with open(filename, "rb") as rf:
    data = pickle.load(rf)
  print(data.keys())

  data['name'] = filename
  export_visualization_file(fn_prefix+"_prediction.complex", data)
  
  if not args.no_nms:
    NMS(data)
  flag_valid = programming_ilp(data)

  if not flag_valid:
    f = open(os.path.join("", fn_prefix+"_opt.wrong"), "w")
    f.close()
    return
  export_visualization_file(fn_prefix+"_extraction.complex", data)


def process_all():
  folder_name = args.folder
  allfs = os.listdir(folder_name)
  # flag_process = 0
  tasks = []
  for f in allfs:
    if f.endswith('.pkl'):
      if args.skip and os.path.exists(os.path.join(folder_name, f.replace('_prediction.pkl', '_extraction.complex'))):
        continue
      filename = os.path.join(folder_name, f)
      tasks.append(filename)

  #no parallel
  if not flag_parallel:
    for i in range(len(tasks)):
      process_one_file(tasks[i])  
    return

  print('#tasks: ', len(tasks))
  #batch processing
  with Pool(num_parallel) as p:
    p.map(process_one_file, tasks)

if __name__ == '__main__':
  process_all()