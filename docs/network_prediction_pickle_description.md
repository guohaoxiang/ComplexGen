### pickle file description

Each pickle file can be viewed as a dictionary containing the following information:

**'patches'**: 
+ 'gt': ground truth information
  + 'type': type labels ('Cylinder': 0, 'Torus': 1, 'BSpline': 2, 'Plane': 3, 'Cone': 4, 'Sphere': 5).
  + 'points': list of length _num\_patches_, each item stores the 20x20 grid points on the patch.
+ 'prediction': network prediction
  + 'valid\_prob': validness probability.
  + 'points': 20x20 grid points of all patches.
  + 'type\_prob': type labels.
  + 'closed_prob': closeness probability along u-direction.
  + 'patch_matching_indices': correspondence between predicted and ground-truth patches, stored in _(predicted\_patch\_id, ground\_truth\_patch\_id)_.

**curves**:
+ 'gt': ground truth information
  + 'type': type labels ('Circle': 0, 'BSpline': 1, 'Line': 2, 'Ellipse': 3).
  + 'points': 34 sample points of all curves.
  + 'is_closed': whether the curve is closed.
+ 'prediction': network prediction
  + 'valid\_prob': validness probability.
  + 'points': 34 sampled points of all curves.
  + 'type\_prob': type labels.
  + 'closed_prob': closeness probability.
  + 'curve_matching_indices': correspondence between predicted and ground-truth curves, stored in _(predicted\_curve\_id, ground\_truth\_curve\_id)_.

**'corners'**: 
+ 'gt': ground truth position of all corners
+ 'prediction': network prediction
  + 'valid\_prob': validness probability.
  + 'position': position of all corners.
  + 'corner_matching_indices': correspondence between predicted and ground-truth corners, stored in _(predicted\_corner\_id, ground\_truth\_corner\_id)_.


**'patch_curve_similarity'**: patch-to-curve adjacent matrix.

**'patch_corner_similarity'**: patch-to-corner adjacent matrix.

**'curve_corner_similarity'**: curve-to-corner adjacent matrix.
