import os
from vis_util import *
from write_obj import *
from load_obj import *
from load_ply import *
import numpy as np
import json

import argparse

parser = argparse.ArgumentParser(description='All in one command to get virtual scanner result')
parser.add_argument('-i', type=str, required = True, help='input file')

args = parser.parse_args()

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


def load_complex_file(fn):
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
    pred_data['pred_corner_position'] = np.zeros([1, n_corner, 3])
    pred_data['pred_logits'] = np.zeros([1, n_corner, 2])
    pred_data['pred_logits'][0][:,0] = 1.0

    for i in range(n_corner):
      line = lines[lineid].split()
      for j in range(3):
        pred_data['pred_corner_position'][0][i][j] = float(line[j])
      lineid += 1

    #curve
    pred_data['pred_curve_logits'] = np.zeros([1, n_curve, 2])
    pred_data['pred_curve_logits'][0][:,0] = 1
    pred_data['pred_curve_type'] = np.zeros([1, n_curve, 4]) 
    
    pred_data['closed_curve_logits'] = np.zeros([1, n_curve, 2])
    pred_data['pred_curve_points'] = np.zeros([1,n_curve, n_curve_sample, 3])
    for i in range(n_curve):
      line = lines[lineid].split()
      for j in range(n_curve_sample):
        for k in range(3):
          pred_data['pred_curve_points'][0][i][j][k] = float(line[2 + 3 * j + k])
      pred_data['pred_curve_type'][0][i][curve_type_to_id(line[0])] = 1.0
      if float(line[1]) > 0.5:
        pred_data['closed_curve_logits'][0][i][1] = 1.0
      else:
        pred_data['closed_curve_logits'][0][i][0] = 1.0

      lineid += 1
    
    #patch
    # print('corner: {} curve: {} patch: {}'.format(n_corner, n_curve, n_patch))
    pred_data['pred_patch_points'] = np.zeros([1, n_patch, n_patch_sample, 3])
    pred_data['pred_patch_logits'] = np.zeros([1, n_patch, 2])
    pred_data['pred_patch_logits'][0][:,0] = 1
    pred_data['pred_patch_type'] = np.zeros([1,n_patch, 6])
    pred_data['closed_patch_logits'] = np.zeros([1, n_patch, 2]) 
    

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

    return pred_data
  
def load_yaml_json_file(fn):
    n_curve_sample = 34
    n_patch_sample = 100
    with_curve_corner = True
    flag_with_param = False
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
    
    
    pred_data['pred_patch_points'] = np.zeros([1, n_patch, n_patch_sample, 3])
    pred_data['pred_patch_logits'] = np.zeros([1, n_patch, 2])
    pred_data['pred_patch_logits'][0][:,0] = 1
    pred_data['pred_patch_type'] = np.zeros([1,n_patch, 6])
    pred_data['closed_patch_logits'] = np.zeros([1, n_patch, 2])

    pred_data['pred_patch_with_param'] = np.zeros([1, n_patch])
    if flag_with_param:
      pred_data['pred_patch_param'] = np.zeros([1, n_patch, 7])
      pred_data['pred_patch_type_name'] = ['' for i in range(n_patch)]

    for i in range(n_patch):
      for j in range(n_patch_sample):
        for k in range(3):
          # pred_data['pred_patch_points'][0][i][j][k] = float(line[1 + 3 * j + k])
          pred_data['pred_patch_points'][0][i][j][k] = patches[i]['grid'][3 * j + k]
      pred_data['pred_patch_type'][0][i][patch_type_to_id(patches[i]['type'])] = 1.0
      if patches[i]['u_closed']: #ignore v_closed
        # pred_data['closed_patch_logits'][0][i][0] = 1.0
        pred_data['closed_patch_logits'][0][i][1] = 1.0
      else:
        # pred_data['closed_patch_logits'][0][i][0] = 0.0
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
      pred_data['pred_corner_position'] = np.zeros([1, n_corner, 3])
      pred_data['pred_logits'] = np.zeros([1, n_corner, 2])
      pred_data['pred_logits'][0][:,0] = 1.0
      for i in range(n_corner):
        for j in range(3):
          pred_data['pred_corner_position'][0][i][j] = info['corners'][i]['pts'][j]
      
      n_curve = 0
      if 'curves' in info and info['curves'] != None:
          n_curve = len(info['curves'])
      pred_data['pred_curve_logits'] = np.zeros([1, n_curve, 2])
      pred_data['pred_curve_logits'][0][:,0] = 1
      #type not set yet
      pred_data['pred_curve_type'] = np.zeros([1, n_curve, 4]) 
      
      pred_data['closed_curve_logits'] = np.zeros([1, n_curve, 2]) #not set yet
      pred_data['pred_curve_points'] = np.zeros([1,n_curve, n_curve_sample, 3])
      for i in range(n_curve):
        for j in range(n_curve_sample):
          for k in range(3):
            # pred_data['pred_curve_points'][0][i][j][k] = float(line[2 + 3 * j + k])
            pred_data['pred_curve_points'][0][i][j][k] = info['curves'][i]['pts'][3 * j + k]
        pred_data['pred_curve_type'][0][i][curve_type_to_id(info['curves'][i]['type'])] = 1.0
        if info['curves'][i]['closed']:
        #   pred_data['closed_curve_logits'][0][i][0] = 0.0
          pred_data['closed_curve_logits'][0][i][1] = 1.0

        else:
        #   pred_data['closed_curve_logits'][0][i][1] = 1.0
          pred_data['closed_curve_logits'][0][i][0] = 1.0

      pred_data['patch2curve'] = info['patch2curve']
      pred_data['curve2corner'] = info['curve2corner']
      pred_data['patch2corner'] = info['patch2corner'] #added 1224
    
    return pred_data

def complexjson_to_obj(fn):
    tmp_pth = r'sphere_94.obj'
    tmp_mesh = load_obj_simple(tmp_pth)
    tmp_mesh.vertices = tmp_mesh.vertices / 0.7
    suffix = ''
    if fn.endswith('.complex'):
        pred_data = load_complex_file(fn)
        suffix = '.complex'
    else:
        pred_data = load_yaml_json_file(fn)
        suffix = '.json'
    
    output_fn = fn.replace(suffix, '.obj')
    #get patch 
    allverts_group = []
    allfaces_group = []
    allmtl_group = []
    all_group_name = []
    patch_pts = pred_data['pred_patch_points'][0]
    patch_close_logits = pred_data['closed_patch_logits'][0]
    counter = 0
    #patches
    for gid in range(len(patch_pts)):
        all_group_name.append('patch{}'.format(gid))
        allmtl_group.append('m{}'.format(gid))
        allverts_group.append(np.transpose(patch_pts[gid].reshape(20,20,3), axes=(1,0,2)).reshape(-1,3))
        faces = gen_cylinder_quads(20, 20, counter, flag_xclose = patch_close_logits[gid][0] < patch_close_logits[gid][1])
        allfaces_group.append(faces)
        counter += allverts_group[-1].shape[0]

    #curves
    curve_pts = pred_data['pred_curve_points'][0]
    curve_close_logits = pred_data['closed_curve_logits'][0]
    curve_verts = []
    curve_faces = []

    for cid in range(len(curve_pts)):
        curve_verts = []
        curve_faces = []
        for i in range(len(curve_pts[cid]) - 1):
            cur_verts, cur_faces = gen_cylinder_from_two_points(curve_pts[cid][i], curve_pts[cid][i + 1], counter)
            if (len(cur_faces) > 0):
              curve_verts.append(cur_verts)
              curve_faces += cur_faces
              counter += cur_verts.shape[0]

        if curve_close_logits[cid][0] < curve_close_logits[cid][1]:
            cur_verts, cur_faces = gen_cylinder_from_two_points(curve_pts[cid][-1], curve_pts[cid][0], counter)
            if (len(cur_faces) > 0):
              curve_verts.append(cur_verts)
              curve_faces += cur_faces
              counter += cur_verts.shape[0]

        all_group_name.append('curve{}'.format(cid))
        allmtl_group.append('cylinder')
        curve_verts = np.concatenate(curve_verts)
        allverts_group.append(curve_verts)
        allfaces_group.append(curve_faces)

    #corner
    corner_pts = pred_data['pred_corner_position'][0]
    for i in range(len(corner_pts)):
        cur_verts, cur_faces = gen_sphere_from_point(corner_pts[i], tmp_mesh, counter)
        counter += cur_verts.shape[0]
        all_group_name.append('corner{}'.format(i))
        allmtl_group.append('sphere')
        allverts_group.append(cur_verts)
        allfaces_group.append(cur_faces)
    write_obj_grouped(output_fn, allverts_group, allfaces_group, allmtl_group, all_group_name)


if __name__ == '__main__':
    fn = args.i
    complexjson_to_obj(fn)
