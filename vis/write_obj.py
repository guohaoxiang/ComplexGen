import numpy as np
import os

def write_obj(fn, verts, faces, mtl_name = 'cylinder'):
    f = open(fn, 'w')
    f.write('mtllib complexgen.mtl\n')
    f.write('usemtl {}\n'.format(mtl_name))
    for i in range(len(verts)):
        f.write('v {} {} {}\n'.format(verts[i][0], verts[i][1], verts[i][2]))

    for i in range(len(faces)):
        f.write('f')
        for j in range(len(faces[i])):
            f.write(' {}'.format(faces[i][j] + 1))
        f.write('\n')

    f.close()

def write_obj_grouped(fn, verts_group, faces_group, mtl_group, all_group_name):
    f = open(fn, 'w')
    f.write('mtllib complexgen.mtl\n')
    assert(len(verts_group) == len(faces_group) and len(faces_group) == len(mtl_group))
    for gid in range(len(verts_group)):
        f.write('g {}\n'.format(all_group_name[gid]))
        f.write('usemtl {}\n'.format(mtl_group[gid]))
        verts = verts_group[gid]
        faces = faces_group[gid]
        for i in range(len(verts)):
            f.write('v {} {} {}\n'.format(verts[i][0], verts[i][1], verts[i][2]))

        for i in range(len(faces)):
            f.write('f')
            for j in range(len(faces[i])):
                f.write(' {}'.format(faces[i][j] + 1))
            f.write('\n')
    f.close()