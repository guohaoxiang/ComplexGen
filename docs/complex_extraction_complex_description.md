### complex file description

Each complex file stores the geometry and topological connectivity of a complex.

**2rd line**: _num\_corners, num\_curves, num\_patchs_.

**Next _num\_corners_ lines**: position of each corners.

**Next _num\_curves_ lines**: curve type, probability of closeness, exact position of 34 points uniformly sampled on the curve.

**Next _num\_patch_ lines**: patch type, exact position of 20x20 points uniformly sampled on the patch.

**Next _num\_curves_ lines**: curve-to-corner adjacent matrix.

**Next _num\_patch_ lines**: patch-to-curve adjacent matrix.

**Next _num\_patch_ lines**: probability of patch closeness along u-direction.
