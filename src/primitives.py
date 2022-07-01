"""
This defines the distance from a geometric primitive. The idea is to
sample points from the ground truth surface and find the distance of
these points from the predicted point cloud.
"""
import copy

import numpy as np
import torch

# from src.fitting_utils import match
# from src.guard import guard_sqrt
# from src.utils import chamfer_distance_single_shape

EPS = np.finfo(np.float32).eps


# class ResidualLoss:
#     """
#     Defines distance of points sampled on a patch with corresponding
#     predicted patch for different primitives. There is a closed form
#     formula for distance from geometric primitives, whereas for splines
#     we use chamfer distance as an approximation.
#     """

#     def __init__(self, reduce=True, one_side=False):
#         cp_distance = ComputePrimitiveDistance(reduce, one_side=one_side)
#         self.routines = {"torus": cp_distance.distance_from_torus,
#                          "sphere": cp_distance.distance_from_sphere,
#                          "cylinder": cp_distance.distance_from_cylinder,
#                          "cone": cp_distance.distance_from_cone,
#                          "plane": cp_distance.distance_from_plane,
#                          "closed-spline": cp_distance.distance_from_bspline,
#                          "open-spline": cp_distance.distance_from_bspline}

#     def residual_loss(self, Points, parameters, sqrt=False):
#         distances = {}
#         for k, v in parameters.items():
#             if v is None:
#                 # degenerate case of primitives that are small
#                 continue
#             dist = self.routines[v[0]](points=Points[k], params=v[1:], sqrt=sqrt)
#             distances[k] = [v[0], dist]
#         return distances


class ComputePrimitiveDistance:
    def __init__(self, reduce=True):
        """
        This defines a differentiable routines that gives
        distance of a point from a surface of a predicted geometric
        primitive.
        # TODO Define closed form distance of point from bspline surface.
        """
        self.reduce = reduce

    def distance_from_torus(self, points, params, sqrt=False):
        """
        Distance of points from the torus
        :param points: N x 3
        :param params: axis: 3 x 1, center: 1 x 3, major_radius \in R+, minor_radius \in R+
        """
        #not used
        axis, center, major_radius, minor_radius = params
        axis = axis.reshape((3, 1)) / torch.norm(axis, p=2)
        center = center.reshape((1, 3))

        center2points = points - center
        z_new = center2points @ axis  # N x 1

        x_new = guard_sqrt(torch.sum(center2points ** 2, 1, keepdim=True) - z_new ** 2)  # N x 1

        # min distance for right circle
        right_dst = (guard_sqrt((x_new - major_radius) ** 2 + z_new ** 2) - minor_radius) ** 2

        # min distance for left circle
        left_dst = (guard_sqrt((x_new + major_radius) ** 2 + z_new ** 2) - minor_radius) ** 2

        distance = torch.min(right_dst, left_dst)
        distance = distance.squeeze()

        if sqrt:
            distance = guard_sqrt(distance)

        if self.reduce:
            distance = torch.mean(distance)
        return distance

    def distance_from_plane(self, points, params, sqrt=False):
        """
        Distance of points from the plane
        :param points: N x 3
        :param params: a: 3 x 1, d \in R
        """
        # a, d = params
        a = params[:3]
        d = params[3]
        a = a.reshape((3, 1))

        #debugging:
        # print('a: {} d: {}'.format(a, d))
        # np.savetxt('plane_debug.xyz', points.detach().cpu().numpy())

        # check for the orientation
        try:
            distance = torch.sum((points @ a - d) ** 2, 1)
        except:
            import ipdb;
            ipdb.set_trace()

        if sqrt:
            distance = torch.sqrt(distance)
        if self.reduce:
            distance = torch.mean(distance)

        # Note that this is distance square
        return distance

    def distance_from_sphere(self, points, params, sqrt=False):
        """
        Distance of points from the sphere
        :param points: N x 3
        :param params: c: 3 x 1, radius \in R
        """
        # center, radius = params
        center = params[:3]
        radius = params[3]
        center = center.reshape((1, 3))
        distance = (torch.norm(points - center, p=2, dim=1) - radius) ** 2
        if sqrt:
            distance = torch.sqrt(distance)

        if self.reduce:
            distance = torch.mean(distance)
        return distance

    def distance_from_cylinder(self, points, params, sqrt=False):
        """
        Distance of points from the cylinder.
        :param points: N x 3
        :param params: axis: 3 x 1, center: 1 x 3, radius \in R
        """
        # axis: 3 x 1, center: 1 x 3
        # axis, center, radius = params
        axis = params[:3]
        center = params[3:6]
        radius = params[6]

        center = center.reshape((1, 3))
        axis = axis.reshape((3, 1))

        v = points - center
        prj = (v @ axis) ** 2

        # this is going negative at some point! fix it. Numerical issues.
        # voilating pythagoras
        dist_from_surface = torch.sum(v * v, 1) - prj[:, 0]
        dist_from_surface = torch.clamp(dist_from_surface, min=1e-5)

        distance = torch.sqrt(dist_from_surface) - radius
        # distance.register_hook(self.print_norm)
        distance = distance ** 2

        if sqrt:
            distance = torch.sqrt(distance)

        if torch.sum(torch.isnan(distance)):
            import ipdb;
            ipdb.set_trace()
        if self.reduce:
            distance = torch.mean(distance)

        return distance

    def print_norm(self, x):
        print("printing norm 2", torch.norm(x))

    def distance_from_cone(self, points, params, sqrt=False):
        # axis: 3 x 1
        # apex, axis, theta = params
        axis = params[:3]
        apex = params[3:6]
        theta = params[6]

        apex = apex.reshape((1, 3))
        axis = axis.reshape((3, 1))

        N = points.shape[0]
        # pi_2 = torch.ones(N).cuda()
        try:
            v = points - apex + 1e-8
        except:
            import ipdb;
            ipdb.set_trace()
        

        # #origin version, not consider reverse axis
        # mod_v = torch.norm(v, dim=1, p=2)
        # alpha_x = (v @ axis)[:, 0] / (mod_v + 1e-7)
        # alpha_x = torch.clamp(alpha_x, min=-.999, max=0.999)

        # # safe gaurding against arc cos derivate going at +1/-1.
        # alpha = torch.acos(alpha_x)

        # dist_angle = torch.clamp(torch.abs(alpha - theta), max=3.142 / 2.0)

        # distance = (mod_v * torch.sin(dist_angle)) ** 2

        #our version, consider reverse axis, might be a little longer than prediction
        axis_proj = v @ axis
        rad_vec = v - axis_proj @ axis.view(1, 3)
        
        mod_rad = torch.norm(rad_vec, dim = 1, p = 2, keepdim=True)
        rad_vec_normalize = rad_vec / (mod_rad + 1e-7)

        target_radius = torch.tan(theta) * axis_proj
        target_radius = torch.abs(target_radius) #always positive
        proj_pts = apex + axis_proj @ axis.view(1,3) + target_radius * rad_vec_normalize

        distance = torch.square(points - proj_pts).sum(-1)
        #our version ends

        if sqrt:
            distance = torch.sqrt(distance)
        if self.reduce:
            distance = torch.mean(distance)
        return distance

        

    # def distance_from_bspline(self, points, params, sqrt=False):
    #     """
    #     This is a rather approximation, where we sample points on the original
    #     bspline surface and store it in bspline_points, and we also sample
    #     points on the predicted bspline surface are store them in `points`
    #     """
    #     # Need to define weighted distance.
    #     bspline_points = params[0][0]
    #     return chamfer_distance_single_shape(bspline_points, points, one_side=self.one_side, sqrt=sqrt,
    #                                          reduce=self.reduce)