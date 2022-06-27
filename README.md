## ComplexGen: CAD Reconstruction by B-Rep Chain Complex Generation ##

<p align="center"> 
<img src="/images/teaser.png" width="900">
</p>

This is the official implementation of the following paper:

Guo H X, Liu S L, Pan H, Liu Y, Tong X, Guo B N. ComplexGen: CAD Reconstruction by B-Rep Chain Complex Generation. _SIGGRAPH 2022_

[Paper](https://haopan.github.io/papers/ComplexGen.pdf) | [Project Page](https://haopan.github.io/complexgen.html)

Abstract: _We view the reconstruction of CAD models in the boundary representation (B-Rep) as the detection of geometric primitives of different orders, i.e., vertices, edges and surface patches, and the correspondence of primitives, which are holistically modeled as a chain complex, and show that by modeling such comprehensive structures more complete and regularized reconstructions can be achieved.
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

The data contains surface points along with normals, and its ground truth B-Rep labels. After extracting the zip file under root directory, the data should be organized as the following structure:
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

### Environment setup with Docker
        
        $ docker pull pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
        $ docker run --runtime=nvidia --ipc=host --net=host -v /path/to/complexgen/:/workspace -t -i pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
        $ cd /workspace
        $ apt-get update && apt-get install libopenblas-dev -y && conda install numpy mkl-include pytorch cudatoolkit=10.1 -c pytorch -y && apt-get install git -y && pip install git+https://github.com/NVIDIA/MinkowskiEngine.git@v0.5.0 --user
        $ cd chamferdist && python setup.py install --user && pip install numba --user && pip install methodtools --user && pip install tensorflow-gpu --user && pip install scipy --user && pip install plyfile --user && pip install trimesh --user && cd ..

To test if the environment is set correctly, run:
        
        $ python train_small.py
        
This command will start the training of ComplexNet on a small dataset with 64 CAD models.

### Testing

To test the trained ComplexNet, please first download the trained weights used in our paper here(to do), and unzip it under the root directory:

```
ComplexGen
│
└─── experiments
    │
    └─── default
    │   │
    |   └─── ckpt
    │       │
    |       └─── *.pth
    └─── ...
```

Then run:

        $ python scripts/test_default.py

You can find the evaluation table (test_statistics.xlsx) and network prediction of each model (\*.pkl) under _ComplexGen/experiments/default/test_obj/_. The description of each pickle file (\*.pkl) can be found [here](docs/network_prediction_pickle_description.md). 

You can also get the visualizable models of corner/curve/patch of some test data by running: 

        $ python scripts/test_default_vis.py

A set of 3D models will be generated under _ComplexGen/experiments/default/vis_test/_ which can be visualized using 3D softwares like [MeshLab](https://www.meshlab.net/).

We also provided the forwarded pickle file here (todo). If you want to use it, please download and unzip it under the root directory.

### Training

If you want to train ComplexNet from scratch, run:

        $ python scripts/train_default.py

By default, the ComplexNet is trained on a server with 8 V100 GPUs.
You can change the numder of GPUs by setting the _--gpu_ flag in scripts/train_default, and change batch size by setting the _batch_size_ flag.
The training takes about 3 days to converge. 


## Phase 2: Complex extraction

### Environment setup

gurobi

license


## Phase 3: Geometric refinement

### Environment setup

libigl

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
This implementation takes [DETR](https://github.com/facebookresearch/detr) and [Geometric Tools](https://github.com/davideberly/GeometricTools) as references. We thank the authors for their excellent work.
