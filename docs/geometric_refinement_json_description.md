### JSON file description

Each json file can be viewed as a dictionary containing the following information, and can be visualized following [here](https://github.com/guohaoxiang/ComplexGen#visualization)

**'patches'**: a list of patch entries, each patch entry stores:
+ 'type': geometric type of the patch, including 'Plane', 'Sphere', 'Cylinder', 'Cone', 'Torus', 'BSpline'
+ 'u_dim'/'v_dim': grid dimension of u/v direction (default: 20/20)
+ 'u_closed'/'v_closed': whether the patch is closed along the u/v direction
+ 'with_param': whether the entry stores the exact parametric surface information
+ 'params': an array of length 7, stores the exact parameters of the surface
  + If the patch is a 'Plane' of form _dot(normal, x) - d = 0_: _params[0:3]_ stores the normal direction, _params[3] = d_
  + If the patch is a 'Sphere': _params[0:3] = center, params[3] = radius_
  + If the patch is a 'Cylinder': _params[0:3] = axis\_direction, params[3:6] = point\_on\_axis, params[6] = radius_
  + If the patch is a 'Cone':  _params[0:3] = axis\_direction, params[3:6] = apex, params[6] = radius_
+ 'grid': exact position of the u_dim * v_dim grid points on the patch

**'curves'**: a list of curve entries, each curve entry stores:
+ 'type': geometric type of the curve, including 'Line', 'Circle', 'Ellipse', 'BSpline'
+ 'closed': whether the curve is closed
+ 'pts': exact position of 34 points uniformly sampled on the curve

**'corners'**: a list of corner entries, each corner entry stores its position

**'patch2curve'**: patch-to-curve adjacent matrix

**'patch2corner'**: patch-to-corner adjacent matrix

**'curve2corner'**: curve-to-corner adjacent matrix
