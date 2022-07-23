import os
import numpy as np

from write_obj import *

def load_curve_obj(fn):
    #return:
    #verts: n x3,  numpy array
    #edges: lists of lists, contains the vert id that forms a curve (starting from 0)
    verts = []
    edges = []
    f = open(fn, 'r')
    lines = f.readlines()
    f.close()

    for line in lines:
        line = line.strip().split(' ')
        if line[0] == 'v':
            point = []
            for i in range(3):
                point.append(float(line[i + 1]))
            verts.append(point)
        elif line[0] == 'l':
            edge = []
            for i in range(len(line) - 1):
                edge.append(int(line[i + 1]) - 1)
            edges.append(edge)
    
    verts = np.array(verts)

    return verts, edges

def load_curve_obj_group(fn):
    #return:
    #verts: n x3,  numpy array
    #edges: lists of lists, contains the vert id that forms a curve (starting from 0)
    verts = []
    edges_group = []
    f = open(fn, 'r')
    lines = f.readlines()
    f.close()

    for line in lines:
        line = line.strip().split(' ')
        if line[0] == 'v':
            point = []
            for i in range(3):
                point.append(float(line[i + 1]))
            verts.append(point)
        elif line[0] == 'l':
            edges = []
            edge = []
            for i in range(len(line) - 1):
                edge.append(int(line[i + 1]) - 1)
            edges.append(edge)
            edges_group.append(edges)
    
    verts = np.array(verts)

    return verts, edges_group

class mytrimesh:
  def __init__(self, verts, faces):
    self.vertices = verts
    self.faces = faces

class trimesh_group:
  def __init__(self, verts_group, faces_group, mtl_group, name_group):
    self.verts_group = verts_group
    self.faces_group = faces_group
    self.mtl_group = mtl_group
    self.name_group = name_group

  def write_obj(self, fn):
    write_obj_grouped(fn, self.verts_group, self.faces_group, self.mtl_group, self.name_group)


def load_obj_simple(filename):
  vertex_list = []
  face_list = []
  #assume only vertex and position contained in obj file
  #triangle mesh
  with open(filename,"r") as rf:
    for line in rf:
      if(line[0] == 'v' and line[1] == ' '):
        #vertex
        arr = line.split()
        vertex_list.append([float(arr[1]), float(arr[2]), float(arr[3])])
      elif(line[0] == 'f' and line[1] == ' '):
        #face
        arr = line.split()
        face_list.append([int(arr[1].split("/")[0]), int(arr[2].split("/")[0]), int(arr[3].split("/")[0])])
  #return np.array(vertex_list, dtype=np.float32), np.array(face_list, dtype=np.int32)
  return mytrimesh(np.array(vertex_list, dtype=np.float32), np.array(face_list, dtype=np.int32)-1)

def load_obj_group(filename): #should be a face model
  verts_group = []
  faces_group = []
  mtl_group = []
  name_group = []

  with open(filename,"r") as rf:
    lines = rf.readlines()
    line_iter = 0
    while line_iter < len(lines):
      line = lines[line_iter].split(' ')
      if line[0] == 'g':
        #group found
        name_group.append(line[1].strip())
        line_iter += 1
        line = lines[line_iter].split(' ')
        mtl_group.append(line[1].strip())
        #verts and faces
        line_iter += 1
        vertex_list = []
        face_list = []
        while line_iter < len(lines):
          line = lines[line_iter].split(' ')
          if(line[0] == 'v'):
            arr = line
            vertex_list.append([float(arr[1]), float(arr[2]), float(arr[3])])
            line_iter += 1
          elif (line[0] == 'f'):
            arr = line
            face_list.append([int(arr[1].split("/")[0]), int(arr[2].split("/")[0]), int(arr[3].split("/")[0])])
            line_iter += 1
            # print(face_list[-1])
          else:
            break
        verts_group.append(np.array(vertex_list, dtype=np.float32))
        # np.savetxt('test.txt', np.array(face_list, dtype=np.int32))
        faces_group.append(np.array(face_list, dtype=np.int32) - 1)


      else:
        line_iter += 1

  return trimesh_group(verts_group, faces_group, mtl_group, name_group)


def merge_group_mesh(mesh_group_list):
  verts_group = []
  faces_group = []
  mtl_group = []
  name_group = []
  cur_vert_id = 0
  for i in range(len(mesh_group_list)):
    verts_group += mesh_group_list[i].verts_group
    # faces_group += mesh_group_list[i].faces_group + cur_vert_id
    faces_group += [fs + cur_vert_id for fs in mesh_group_list[i].faces_group]
    mtl_group += mesh_group_list[i].mtl_group
    name_group += mesh_group_list[i].name_group
    cur_vert_id += sum([len(vs) for vs in mesh_group_list[i].verts_group])

  return trimesh_group(verts_group, faces_group, mtl_group, name_group)