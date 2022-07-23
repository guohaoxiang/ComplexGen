
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm

def gen_cylinder_quads(x_split, y_split, counter = 0, flag_xclose = True):
    #x is closed
    faces = []
    for j in range(y_split - 1):
        for i in range(x_split - 1):
            oneface = []
            oneface.append(counter + j * x_split + i)
            oneface.append(counter + (j + 1) * x_split + i)
            oneface.append(counter + (j + 1) * x_split + i + 1)
            oneface.append(counter + j * x_split + i + 1)
            faces.append(oneface)
        
        if flag_xclose:
            oneface = []
            oneface.append(counter + j * x_split + x_split - 1)
            oneface.append(counter + (j + 1) * x_split + x_split - 1)
            oneface.append(counter + (j + 1) * x_split)
            oneface.append(counter + j * x_split)
            faces.append(oneface)
    
    return faces

def gen_cylinder_tris(x_split, y_split, counter = 0, flag_xclose = True):
    #x is closed
    faces = []
    for j in range(y_split - 1):
        for i in range(x_split - 1):
            oneface = []
            oneface.append(counter + j * x_split + i)
            oneface.append(counter + (j + 1) * x_split + i)
            oneface.append(counter + (j + 1) * x_split + i + 1)
            # oneface.append(counter + j * x_split + i + 1)
            faces.append(oneface)
            oneface = []
            oneface.append(counter + j * x_split + i)
            # oneface.append(counter + (j + 1) * x_split + i)
            oneface.append(counter + (j + 1) * x_split + i + 1)
            oneface.append(counter + j * x_split + i + 1)
            faces.append(oneface)
        
        if flag_xclose:
            oneface = []
            oneface.append(counter + j * x_split + x_split - 1)
            oneface.append(counter + (j + 1) * x_split + x_split - 1)
            oneface.append(counter + (j + 1) * x_split)
            # oneface.append(counter + j * x_split)
            faces.append(oneface)
            oneface = []
            oneface.append(counter + j * x_split + x_split - 1)
            # oneface.append(counter + (j + 1) * x_split + x_split - 1)
            oneface.append(counter + (j + 1) * x_split)
            oneface.append(counter + j * x_split)
            faces.append(oneface)
    
    return faces

def gen_cylinder_from_two_points(p0, p1, counter = 0, x_split = 20, y_split = 2, R = 0.006, extend_radio = 1.1):
    v = p1 - p0
    #find magnitude of vector
    mag = norm(v)
    if mag < 1e-5:
        print('curve end points too close')
        return [],[]
    center = (p0 + p1) / 2.0
    p1 = center + extend_radio * v / 2.0
    p0 = center - extend_radio * v / 2.0
    #unit vector in direction of axis
    v = v / mag
    mag = mag * extend_radio
    #make some vector not in the same direction as v
    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])
    #make vector perpendicular to v
    n1 = np.cross(v, not_v)
    #normalize n1
    n1 /= norm(n1)
    #make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    #surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(0, mag, y_split)
    theta = np.linspace(0, 2 * np.pi, x_split)
    #use meshgrid to make 2d arrays
    # t, theta = np.meshgrid(t, theta)

    theta, t = np.meshgrid(theta, t)

    #generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]

    # faces = gen_cylinder_quads(x_split, y_split, counter)
    faces = gen_cylinder_tris(x_split, y_split, counter)


    verts = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)], 1)

    return verts, faces


def gen_sphere_from_point(p0, tmp_mesh, counter, R = 0.02):
    # verts = tmp_mesh.vertices / 0.7
    verts = tmp_mesh.vertices
    faces = tmp_mesh.faces

    return verts * R + p0, faces + counter