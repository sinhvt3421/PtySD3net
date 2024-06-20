# Table of Contents

* [Introduction](#introduction)
* [PtySD3Net Framework](#ptysd3net-framework)
* [Installation](#installation)
<!-- * [Usage](#usage)
* [Datasets](#datasets)
* [References](#references) -->

<a name="introduction"></a>

# Introduction
This repository is the official implementation of [PtySD3Net: Single-Shot Dynamic Phase Retrieval using 3D Temporal Convolution-Based Deep Diffraction Imaging Networks]().

Please cite us as

```
@article{Vu2024,

}
```

<a name="DeepAt-framework"></a>

# PtySD3Net framework


# Installation

Firstly, create a conda environment to install the package, for example:
```
conda create -n test python==3.9
source activate test
```

### Optional GPU dependencies

For hardwares that have CUDA support, the <b>tensorflow version with gpu options</b> should be installed. Please follow the installation from https://www.tensorflow.org/install for more details.

Tensorflow can  also be installed from ```conda``` for simplification settings:
```
conda install -c conda-forge tensorflow-gpu
```

#### Method 1 (directly install from git)
You can install the lastes development version of PtySD3Net from this repo and install using:
```
git clone https://github.com/sinhvt3421/PtySD3net
cd PtySD3net
python -m pip install -e .
```