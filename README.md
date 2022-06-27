## ComplexGen: CAD Reconstruction by B-Rep Chain Complex Generation ##

<p align="center"> 
<img src="/images/teaser.png" width="900">
</p>

This is the official implementation of the following paper:

Guo H X, Liu S L, Pan H, Liu Y, Tong X, Guo B N. ComplexGen: CAD Reconstruction by B-Rep Chain Complex Generation. _SIGGRAPH 2022_

[Paper](https://haopan.github.io/papers/ComplexGen.pdf) | [Project Page](https://haopan.github.io/complexgen.html)

Abstract: _We view the reconstruction of CAD models in the boundary representation (B-Rep) as the detection of geometric primitives of different orders, i.e. vertices, edges and surface patches, and the correspondence of primitives, which are holistically modeled as a chain complex, and show that by modeling such comprehensive structures more complete and regularized reconstructions can be achieved.
We solve the complex generation problem in two steps.
First, we propose a novel neural framework that consists of a sparse CNN encoder for input point cloud processing and a tri-path transformer decoder for generating geometric primitives and their mutual relationships with estimated probabilities.
Second, given the probabilistic structure predicted by the neural network, we recover a definite B-Rep chain complex by solving a global optimization maximizing the likelihood under structural validness constraints and applying geometric refinements.
Extensive tests on large scale CAD datasets demonstrate that the modeling of B-Rep chain complex structure enables more accurate detection for learning and more constrained reconstruction for optimization, leading to structurally more faithful and complete CAD B-Rep models than previous results._

<p align="center"> 
<img src="/images/pipeline.png" width="1000">
</p>

The pipeline contains 3 main phases, we will show how to run the code for each phase, and provide the corresponding checkpoint/data.

## Data downloading
We provide the pre-processed ABC dataset here (to do). You can find the details of pre-processing pipelines in the [supplemental material](https://haopan.github.io/data/ComplexGen_supplemental.zip) of our paper.

The data contains surface points along with normals, and its ground truth B-Rep labels. The data should be organized as the following structure:
```
ComplexGen
│
└─── data
    │
    └─── default
    │   │
    |   └─── train
    │   │
    |   └─── eval
    │   │
    |   └─── test
    |   |   
    |   └─── test_point_clouds
    |         
    └─── noise_002
    │   │
    |   └─── ...
    └─── noise_005
    │   │
    |   └─── ...
    └─── partial
        │
        └─── ...
```

Here _noise_002_ and _noise_005_ means noisy point clouds with normal-distribution-perturbation of mean value _0.02_ and _0.05_ respectively.

## Phase 1: ComplexNet prediction

        $ cd cext
        $ mkdir build
        $ cd build
        $ cmake ..
        $ make

## Phase 2: Complex extraction

## Phase 3: Geometric refinement

## Test on your own point cloud
If you want to use our trained model to test on your own point cloud, please follow these steps:

## Citation

If you use our code for research, please cite our paper:
```
@article{GuoComplexGen2022,
    author = {Haoxiang Guo and Shilin Liu and Hao Pan and Yang Liu and Xin Tong and Baining Guo},
    title = {ComplexGen: CAD Reconstruction by B-Rep Chain Complex Generation},
    year = {2022},
    issue_date = {July 2022},
    publisher = {Association for Computing Machinery},
    volume = {41},
    number = {4},
    url = {https://doi.org/10.1145/3528223.3530078},
    doi = {10.1145/3528223.3530078},
    journal = {ACM Trans. Graph. (SIGGRAPH)},
    month = jul,
    articleno = {129},
    numpages = {18}
}
```

## License

MIT Licence

## Contact

Please contact us (Haoxiang Guo guohaoxiangxiang@gmail.com) if you have any question about our implementation.

## Acknowledgement
This implementation takes [DETR](https://github.com/facebookresearch/detr) and [Geometric Tools](https://github.com/davideberly/GeometricTools) as a reference. We thank the authors for their excellent work.
