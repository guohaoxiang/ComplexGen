## ComplexGen: CAD Reconstruction by B-Rep Chain Complex Generation ##

This is the official implementation of the following paper:

Guo H X, Liu S L, Pan H, Liu Y, Tong X, Guo B N. ComplexGen: CAD Reconstruction by B-Rep Chain Complex Generation. _SIGGRAPH 2022_

[Paper](https://haopan.github.io/papers/ComplexGen.pdf) | [Project Page](https://haopan.github.io/complexgen.html)

Abstract: _We view the reconstruction of CAD models in the boundary representation (B-Rep) as the detection of geometric primitives of different orders, i.e. vertices, edges and surface patches, and the correspondence of primitives, which are holistically modeled as a chain complex, and show that by modeling such comprehensive structures more complete and regularized reconstructions can be achieved.
We solve the complex generation problem in two steps.
First, we propose a novel neural framework that consists of a sparse CNN encoder for input point cloud processing and a tri-path transformer decoder for generating geometric primitives and their mutual relationships with estimated probabilities.
Second, given the probabilistic structure predicted by the neural network, we recover a definite B-Rep chain complex by solving a global optimization maximizing the likelihood under structural validness constraints and applying geometric refinements.
Extensive tests on large scale CAD datasets demonstrate that the modeling of B-Rep chain complex structure enables more accurate detection for learning and more constrained reconstruction for optimization, leading to structurally more faithful and complete CAD B-Rep models than previous results._
