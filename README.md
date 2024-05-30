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
We provide the pre-processed ABC dataset used for training and evaluating ComplexNet, you can download it from [BaiduYun](https://pan.baidu.com/s/1PStVn2h_kkKtYsc-LYF7sQ?pwd=asdf) or [OneDrive](https://1drv.ms/u/s!Ar3e2GVr5NQN9WluD3_SdPUSBNRN?e=uRgd2B), which can be extracted by [7-Zip](https://www.7-zip.org/). You can find the details of pre-processing pipelines in the [supplemental material](https://haopan.github.io/data/ComplexGen_supplemental.zip) of our paper.

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
    |   └─── val
    │   │
    |   └─── test
    |   |   
    |   └─── test_point_clouds
    |        
    └─── partial
        │
        └─── ...
```

<!-- Here _noise_002_ and _noise_005_ means noisy point clouds with normal-distribution-perturbation of mean value _0.02_ and _0.05_ respectively. -->

**\[Optional\]** You can also find the output of each phase from [BaiduYun](https://pan.baidu.com/s/1vO0nTSBbCw52EWUDZI7X4g?pwd=asdf) or [OneDrive](https://1drv.ms/u/s!Ar3e2GVr5NQN9WoahuXFZ2eiVS3Y?e=4nQfIi). For each test model, there will be 4 or 5 outputs:
```
*_input.ply: Input point cloud
*_prediction.pkl: Output of 'ComplexNet prediction' phase
*_prediction.complex: Visualizable file for *_prediction.pkl, elements with valid probability larger than 0.3 are kept.
*_extraction.complex: Output of 'complex extraction' phase
*_geom_refine.json: Output of 'geometric refinement' phase, which is also the final output.
```
The description and visualization of each file type can be found in [pickle description](docs/network_prediction_pickle_description.md), [complex description](docs/complex_extraction_complex_description.md) and [json description](docs/geometric_refinement_json_description.md). If you want to directly evaluate the provided output data of ComplexGen, please put the extracted _experiments_ folder under root folder _ComplexGen_, and conduct [Environment setup](https://github.com/guohaoxiang/ComplexGen#environment-setup-with-docker) and [Evaluation](https://github.com/guohaoxiang/ComplexGen#evaluation)

## Phase 1: ComplexNet prediction

### Environment setup with Docker
        
        $ docker pull pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
        $ docker run --runtime=nvidia --ipc=host --net=host -v /path/to/complexgen/:/workspace -t -i pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
        $ cd /workspace
        $ apt-get update && apt-get install libopenblas-dev -y && conda install numpy mkl-include pytorch cudatoolkit=10.1 -c pytorch -y && apt-get install git -y && pip install git+https://github.com/NVIDIA/MinkowskiEngine.git@v0.5.0 --user
        $ cd chamferdist && python setup.py install --user && pip install numba --user && pip install methodtools --user && pip install tensorflow-gpu --user && pip install scipy --user  && pip install rtree --user && pip install plyfile --user && pip install trimesh --user && cd ..


**\[Note\]:** If the 'apt-get update' gets error. To solve this problem, you can firstly run command 'rm /etc/apt/sources.list.d/cuda.list' (details shown in NVIDIA/nvidia-docker#619).

To test if the environment is set correctly, run:
        
        $ ./scripts/train_small.sh
        
This command will start the training of ComplexNet on a small dataset with 64 CAD models.

### Testing

To test the trained ComplexNet, please first download the trained weights used in our paper from [BaiduYun](https://pan.baidu.com/s/1fvwURG1FWjazvQpVVASwMg?pwd=asdf) or [OneDrive](https://1drv.ms/u/s!Ar3e2GVr5NQN9Wyk28t7vqCWQZnO?e=3OZapR), and unzip it under the root directory:

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

        $ ./scripts/test_default.sh

You can find network prediction of each model (\*.pkl) under _ComplexGen/experiments/default/test_obj/_. The description of each pickle file (\*.pkl) can be found [here](docs/network_prediction_pickle_description.md). 

You can also get the visualizable models of corner/curve/patch of some test data by running: 

        $ ./scripts/test_default_vis.sh

A set of 3D models will be generated under _ComplexGen/experiments/default/vis_test/_ which can be visualized using 3D softwares like [MeshLab](https://www.meshlab.net/).

<!-- We also provided the forwarded pickle file here (todo). If you want to use it, please download and unzip it under the root directory. -->

### Training

If you want to train ComplexNet from scratch, run:

        $ ./scripts/train_default.sh

By default, ComplexNet is trained on a server with 8 V100 GPUs.
You can change the numder of GPUs by setting the _--gpu_ flag in _./scripts/train\_default.sh_, and change batch size by setting the _batch_size_ flag.
The training takes about 3 days to converge. 


## Phase 2: Complex extraction

### Environment setup

        $ pip install gurobipy==9.1.2 && pip install Mosek && pip install sklearn

Note that you need also mannully setup [Gurobi license](https://support.gurobi.com/hc/en-us/articles/360059842732).

To conduct complex extraction, run:

        $ ./scripts/extraction_default.sh

A set of complex file will be generated under _ComplexGen/experiments/default/test_obj/_. The description and visualization of complex file can be found [here](docs/complex_extraction_complex_description.md). As the average extraction time for each model is 10 minutes, we recommend you to conduct complex extraction on a multi-thread cpu server. To do this, just set _flag\_parallel_ as _True_ and _num\_parallel_ as half of the number of available threads in _ComplexGen/PostProcess/complex\_extraction.py_.

## Phase 3: Geometric refinement

Code of this phase can be compiled only under Windows. If you want to build it under Linux, please follow the Chinese instructions [here](https://azaan-zheng.github.io/2023/02/12/CAD%E5%AE%9E%E4%BD%93%E5%BB%BA%E6%A8%A1-ComplexGen%E5%87%A0%E4%BD%95%E4%BC%98%E5%8C%96%E9%85%8D%E7%BD%AE/) or [here](https://github.com/Mingyu-Zheng/mingyu-zheng.github.io/blob/master/_posts/geometric-modeling/2023-2-12-%E3%80%8CCAD%E5%AE%9E%E4%BD%93%E5%BB%BA%E6%A8%A1%E3%80%8DComplexGen%E5%87%A0%E4%BD%95%E4%BC%98%E5%8C%96%E9%85%8D%E7%BD%AE.md).
<!-- as I fail to compile and link [clapack](https://netlib.org/clapack/) with our project under Linux. Sorry for the inconvenience :-( -->

### Environment setup
libigl and Eigen are needed, you can install them via [vcpkg](https://github.com/microsoft/vcpkg)

        $ vcpkg.exe integrate install
        $ vcpkg.exe install libigl
        $ vcpkg.exe install eigen3

### Compile and build
The C++ project can be generated with CMake:

        $ cd PATH_TO_COMPLEXGEN\GeometricRefine
        $ mkdir build
        $ cd build
        $ cmake ..

Then you can build _GeometricRefine.sln_ with Visual Studio. After that, you'll find _GeometricRefine.exe_ under _PATH_TO_COMPLEXGEN/GeometricRefine/Bin_.

To conduct geometric refinement for all models, please first modify _.\scripts\geometric_refine.py_ by setting 'pc_ply_path' as the path containing the input point cloud stored in _.ply_ format, and setting 'complex_path' as the path containing the results of complex extraction, then run:

        $ cd PATH_TO_COMPLEXGEN
        $ python .\scripts\geometric_refine.py

If you are processing noisy/partial data, please replace the second command with:

        $ python .\scripts\geometric_refine.py --noise

You will find the generate json files under 'complex_path'. Description of the generated json file can be found  [here](docs/geometric_refinement_json_description.md)

## Evaluation

The evaluation is conducted under Linux. To evaluate the final output of ComplexGen, run:

        $ ./scripts/eval_default.sh

You can find the metrics of each model and all models in _ComplexGen/experiments/default/test_obj/final_evaluation_geom_refine.xlsx_.

## Visualization

We provide tools for converting our generated complex file or json file to obj files which can be visualized with [MeshLab](https://www.meshlab.net/):
        
        $ cd vis
        $ python gen_vis_result.py -i PATH_TO_COMPLEX/JSON_FILE

Remember to copy _./vis/complexgen.mtl_ to the target folder containing the complex/json file. Corners of the reconstructed B-Rep are shown in yellow, curves in blue and patches in different colors.

<!-- ## Test on your own point cloud -->
<!-- If you want to use our trained model to test on your own point cloud, please follow these steps: -->

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
