# Robust Equivariant Imaging (REI)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2111.12855)
[![GitHub Stars](https://img.shields.io/github/stars/edongdongchen/REI?style=social)](https://github.com/edongdongchen/REI)

[Robust Equivariant Imaging: a fully unsupervised framework for learning to image from noisy and partial measurements](https://arxiv.org/pdf/2111.12855.pdf)

[Dongdong Chen](https://dongdongchen.com), [Julián Tachella](https://tachella.github.io/), [Mike E. Davies](https://www.research.ed.ac.uk/en/persons/michael-davies).

The University of Edinburgh, UK

In CVPR 2022 (oral)


## Background

Deep networks provide state-of-the-art performance in multiple imaging inverse problems ranging from medical imaging to computational photography. However, most existing networks are trained with clean signals which are often hard or impossible to obtain.

#### [Equivariant Imaging (EI)](https://github.com/edongdongchen/EI)

**EI** is the first `self-supervised` learning framework that exploits the `group invariance` resent in signal distributions to learn a reconstruction function from partial measurement data alone. EI is `end-to-end` and `physics-based` learning framework for inverse problems with theoretical guarantees which leverages simple but fundamental priors about natural signals: `symmetry` and `low-dimensionality`. Note given an inverse problem, EI learns the reconstruction function from the measurement data alone, with **NO** need for either multiple forward operators or extra masking measurement data into multiple complementary/overlapped parts. Please find our [blog post](https://tachella.github.io/projects/equivariantimaging/) for a quick introduction of EI.

#### Robust Equivariant Imaging (REI)

* Motivation: while EI results are impressive and solved the challege of learning the nullspace without groundtruth, its performance degrades with increasing measurement noise (see below fig 1). 
* Main idea: we propose to employ `Stein's Unbiased Risk Estimator (SURE)` to obtain a fully unsupervised training loss that is robust to noise, i.e. make SURE to provide an unbiased estimator to the measurement consistency loss of clean measurements. With the SURE loss and the EI objective, we propose a REI framework which can learn to image from noisy partial measurements alone (see the diagram of REI in below fig 2). 
* Performance: REI can obatin considerable performance gains on linear (e.g. MRI, Inpainting) and nonlinear inverse problems (e.g. CT), thereby paving the way for robust unsupervised imaging with deep networks (see below fig 3).

## Run the code

1. Requirements: configure the environment by following: `./environment.yml` to run Inpainting and CT experiments. To run MRI experiments, please install the 'fastmri' package by `pip install fastmri`.

2. Find the implementation of Robust Equivariant Imaging (**REI**):
   * `./rei/closure/rei_end2end.py`: REI for `Guassian` noise (e.g. in 'accelerated MRI' task) and `Poisson` noise (e.g. in 'Inpainting' task)
   * `./rei/closure/rei_end2end_ct.py`: REI for `Mixed Poisson-Gaussian (MPG)` noise model (e.g. in 'low-dose&sparse-view CT' task)

3. Download datasets from the below source and move them under the folders: './dataset/mri', './dataset/Urban100', and './dataset/CT', repectively:
   * fastMRI (only the subset 'Knee MRI'): https://fastmri.med.nyu.edu/
   * Urban100: https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip
   * CT100: https://www.kaggle.com/kmader/siim-medical-images

4. **Train**: run the below scripts to train REI models:
   * run `./demo_scripts/demo_mri.py`, `./demo_scripts/demo_inpainting.py`, `./demo_scripts/demo_ct.py` to train REI for MRI, Inpainting, and CT tasks, respectively.
   * or run `train_bash.sh` to train REI models on all tasks.
   ```
   bash train_bash.sh
   ```
   * run 'demo_test.py' to test the performance (PSNR) of a trained model on a specific task.
   ```
   python3 demo_test.py
   ```

5. **Test**: we provide the trained models used in the paper which can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1Io0quD-RvoVNkCmE36aQYpoouEAEP5pF?usp=sharing). Please put the downloaded folder 'ckp' in the root path. Then evaluate the trained models by running 'demo_test.py' to test the performance (PSNR) of a trained model on a specific task.
```
python3 demo_test.py
``` 

5. to solve a new inverse problem, you may only need to implement your forward model (physics) and specify the path of you dataset.

## Citation
If you use this code for your research, please cite our papers.
  ```
	@inproceedings{chen2021equivariant,
	    title     = {Equivariant Imaging: Learning Beyond the Range Space},
	    author    = {Chen, Dongdong and Tachella, Juli{\'a}n and Davies, Mike E},
	    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
	    month     = {October},
	    year      = {2021},
	    pages     = {4379-4388}}

	@inproceedings{chen2022robust,
	    title     = {Robust Equivariant Imaging: a fully unsupervised framework for learning to image from noisy and partial measurements},
	    author    = {Chen, Dongdong and Tachella, Juli{\'a}n and Davies, Mike E},
	    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	    year      = {2022}}
  ```
