import os
import numpy as np
import struct

def load_ply(filename):
    with open(filename,"r") as rf:
        while(True):
            try:
                line = rf.readline()
            except UnicodeDecodeError:
                return load_ply_binary(filename)
            except:
                raise notImplementedError
            if("end_header" in line):
                break
            if("element vertex" in line):
                arr = line.split()
                num_of_points = int(arr[2])
        
        #print("%d points in ply file" %num_of_points)
        points = np.zeros([num_of_points, 6])
        for i in range(points.shape[0]):
            point = rf.readline().split()
            assert(len(point) == 6)
            points[i][0] = float(point[0])
            points[i][1] = float(point[1])
            points[i][2] = float(point[2])
            points[i][3] = float(point[3])
            points[i][4] = float(point[4])
            points[i][5] = float(point[5])
        return points
        
def load_ply_binary(filename):
    with open(filename,"rb") as rf:
        while(True):
            line = rf.readline()
            if(b"end_header" in line):
                break
            if(b"element vertex" in line):
                arr = line.decode("utf-8").split()
                num_of_points = int(arr[2])
        
        #print("%d points in ply file" %num_of_points)
        points = np.zeros([num_of_points, 6])
        for i in range(points.shape[0]):
            x,y,z,nx,ny,nz = struct.unpack('ffffff', rf.read(24))
            points[i][0] = x
            points[i][1] = y
            points[i][2] = z
            points[i][3] = nx
            points[i][4] = ny
            points[i][5] = nz
        return points

def load_ply_binary_vq(filename):
    with open(filename,"rb") as rf:
        while(True):
            line = rf.readline()
            if(b"end_header" in line):
                break
            if(b"element vertex" in line):
                arr = line.decode("utf-8").split()
                num_of_points = int(arr[2])
        
        #print("%d points in ply file" %num_of_points)
        points = np.zeros([num_of_points, 7])
        for i in range(points.shape[0]):
            x,y,z,nx,ny,nz,vq = struct.unpack('fffffff', rf.read(28))
            points[i][0] = x
            points[i][1] = y
            points[i][2] = z
            points[i][3] = nx
            points[i][4] = ny
            points[i][5] = nz
            points[i][6] = vq
        return points
   
def write_ply(filename, points):
    with open(filename,"w") as wf:
        if(points.shape[1] == 6):
            wf.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nend_header\n" %points.shape[0])
            for i in range(points.shape[0]):
                wf.write("%lf %lf %lf %lf %lf %lf\n" %(points[i][0], points[i][1], points[i][2], points[i][3], points[i][4], points[i][5]))
        else:
            assert(points.shape[1] == 3)
            wf.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n" %points.shape[0])
            for i in range(points.shape[0]):
                wf.write("%lf %lf %lf\n" %(points[i][0], points[i][1], points[i][2]))
    
def save_ply(filename, points):
    return write_ply(filename,points)

def bounding_box(points):
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    print("bounding box (%lf %lf %lf) - (%lf %lf %lf)" %(x.min(), y.min(), z.min(), x.max(), y.max(), z.max()))
    return (x.min(), y.min(), z.min(), x.max(), y.max(), z.max())
    
def points_mean(points):
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    #print("mean points of cloud: (%lf %lf %lf)" %(x.mean(), y.mean(), z.mean()))
    return np.array([x.mean(), y.mean(), z.mean()])

def bounding_sphere(points):
    return points_mean(points), bounding_sphere_length(points)

def bounding_sphere_length(points):
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    sq = (x - x.mean())**2 + (y-y.mean())**2 + (z-z.mean())**2
    #print("center at (%lf %lf %lf) radius = %lf" %(x.mean(), y.mean(), z.mean(), sq.max()**0.5))
    return np.array([x.mean(), y.mean(), z.mean()]),sq.max()**0.5
    
def points_sampling(points, n_target):
    #assert(points.shape[0] >= n_target)
    if(points.shape[0] < n_target):
        return points
    rand_index = np.arange(points.shape[0])
    np.random.shuffle(rand_index)
    sampled_points = points[rand_index[:n_target], :]
    return sampled_points
    
def points_sampling_saliency(points, saliency, n_target):
    assert(points.shape[0] == saliency.shape[0])
    #assert(points.shape[0] >= n_target)
    if(points.shape[0] < n_target):
        return points
    rand_index = np.arange(points.shape[0])
    np.random.shuffle(rand_index)
    sampled_points = points[rand_index[:n_target], :]
    sampled_saliency = saliency[rand_index[:n_target]]
    return sampled_points, sampled_saliency

def normalize_model(points, edge_length=None, mesh_vertices=None):
    #normalize to unit sphere
    if(points.shape[1] == 6):
      with_normal = True
      pos = points[:,:3]
      normals = points[:,3:]
    else:
      assert(points.shape[1] == 3)
      with_normal = False
      pos = points
    mean, radius = bounding_sphere_length(points)
    pos -= mean
    pos /= (radius+0.0001)
    if(mesh_vertices is not None):
      mesh_vertices -= mean
      mesh_vertices /= (radius+0.0001)
    if(with_normal):
      out_points = np.concatenate([pos, normals], axis=1)
    else:
      out_points = pos
    if(edge_length is None and mesh_vertices is None):
      return out_points
    elif(edge_length is not None and mesh_vertices is not None):
      return out_points, edge_length / (radius+0.0001), mesh_vertices
    else:
      assert(mesh_vertices is None)
      return out_points, edge_length / (radius+0.0001)
    