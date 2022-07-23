import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import norm

from write_obj import *

def gen_cylinder_quads(x_split, y_split, counter = 0):
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
        
        oneface = []
        oneface.append(counter + j * x_split + x_split - 1)
        oneface.append(counter + (j + 1) * x_split + x_split - 1)
        oneface.append(counter + (j + 1) * x_split)
        oneface.append(counter + j * x_split)
        faces.append(oneface)
    
    return faces
    

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#axis and radius

x_split = 20
y_split = 2
R = 5


p0 = np.array([1, 3, 2])
p1 = np.array([8, 5, 9])


# p0 = np.array([0,0,0])
# p1 = np.array([0,0,1])
#vector in direction of axis
v = p1 - p0
#find magnitude of vector
mag = norm(v)
#unit vector in direction of axis
v = v / mag
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


np.savetxt('t.txt', t)
np.savetxt('theta.txt', theta)

#generate coordinates for surface
X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]

faces = gen_cylinder_quads(x_split, y_split, 0)

verts = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)], 1)

write_obj('test.obj', verts, faces)

# ax.plot_surface(X, Y, Z)
# #plot axis
# ax.plot(*zip(p0, p1), color = 'red')
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)
# ax.set_zlim(0, 10)
# plt.show()