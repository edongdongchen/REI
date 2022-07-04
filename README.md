# Robust Equivariant Imaging (REI) in PyTorch

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2111.12855)
[![GitHub Stars](https://img.shields.io/github/stars/edongdongchen/REI?style=social)](https://github.com/edongdongchen/REI)

[Robust Equivariant Imaging: a fully unsupervised framework for learning to image from noisy and partial measurements](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Robust_Equivariant_Imaging_A_Fully_Unsupervised_Framework_for_Learning_To_CVPR_2022_paper.pdf)

[Dongdong Chen](https://dongdongchen.com), [Julián Tachella](https://tachella.github.io/), [Mike E. Davies](https://www.research.ed.ac.uk/en/persons/michael-davies).

The University of Edinburgh, UK

In CVPR 2022 (oral)


## Background

Deep networks provide state-of-the-art performance in multiple imaging inverse problems ranging from medical imaging to computational photography. However, most existing networks are trained with clean signals which are often hard or impossible to obtain. This work aims to solve the challenge: **learn the reconstruction function from noisy and partial measurements alone**. Please find our [presentation video](https://www.youtube.com/watch?v=27iWnWEbQvA) for a quick introduction.

#### [Background: Equivariant Imaging (EI)](https://github.com/edongdongchen/EI)

<div align=center><img width="650" src="https://github.com/edongdongchen/REI/blob/main/images/schematic_equivariance.png"></div>

Figure 1: **Equivariant imaging systems.** If the set of signals is invariant to a certain set of transformations, the composition of imaging operator (<img src="https://render.githubusercontent.com/render/math?math=A">) with the reconstruction function (<img src="https://render.githubusercontent.com/render/math?math=f_\theta">) should be `equivariant` to these transformations.

* **EI** is the first `self-supervised` learning framework that exploits the `group invariance` present in signal distributions to learn a reconstruction function from partial measurement data alone (Figure 1). EI is `end-to-end` and `physics-based` learning framework for inverse problems with theoretical guarantees which leverages simple but fundamental priors about natural signals: `symmetry` and `low-dimensionality`. 
* Given an inverse problem, EI learns the reconstruction function with **NO** need for either multiple forward operators or extra masking measurement data into multiple complementary/overlapped parts. `EI is agnostic to neural network architecture`. Please find our [blog post](https://tachella.github.io/projects/equivariantimaging/) and [presentation video](https://www.youtube.com/watch?v=wGxW5bcCdxo) for a quick introduction of EI.

#### Robust Equivariant Imaging (REI)

<div align=center><img width="650" src="https://github.com/edongdongchen/REI/blob/main/images/fig_cvpr_rei.png"></div>

Figure 2: **REI training strategy.** <img src="https://render.githubusercontent.com/render/math?math=x^{(1)}"> represents the estimated image, <img src="https://render.githubusercontent.com/render/math?math=T_g"> is the transformation, while <img src="https://render.githubusercontent.com/render/math?math=x^{(2)}"> and <img src="https://render.githubusercontent.com/render/math?math=x^{(3)}">  represent <img src="https://render.githubusercontent.com/render/math?math=T_gx^{(1)}"> and the estimate of <img src="https://render.githubusercontent.com/render/math?math=x^{(2)}"> from the (noisy) measurements <img src="https://render.githubusercontent.com/render/math?math=\tilde{y} = A (x^{(2)})"> respectively. The `SURE` loss aims to estimate the measurement consistency of clean measurement, `REQ` (robust equivariance) loss is the error (e.g. MSE) between <img src="https://render.githubusercontent.com/render/math?math=x^{(2)}"> and <img src="https://render.githubusercontent.com/render/math?math=x^{(3)}">.


* *Motivation*: while EI results are impressive and successfully solved the challenge of learning to image without groundtruth, its performance degrades with increasing measurement noise (Figure 2). 
* *Main idea*: we propose to employ `Stein's Unbiased Risk Estimator (SURE)` to obtain a fully unsupervised training loss that is robust to noise, i.e. have an unbiased SURE estimator to the measurement consistency loss of clean measurements. With the SURE loss and the EI objective, our proposed REI framework can learn to image from noisy partial measurements alone (Figure 3, Figure 4). 
* *Performance*: REI can obatin considerable performance gains on linear (e.g. MRI, Inpainting) and nonlinear inverse problems (e.g. CT), thereby paving the way for robust unsupervised imaging with deep networks (Figure 4).
* *Remark 1*: while we evaluated REI on the `Gaussian`, `Poisson` and `Mixed Poisson-Gaussian (MPG)` models, SURE can handle many other models including non-exponential ones, see [Raphan et al.](https://www.cns.nyu.edu/pub/lcv/raphan10.pdf) for a detailed list. By this repo, we believe one can implement other noise models accordingly without giant changes.
* *Remark 2*: `(R)EI is agnostic to neural network architecture` -- one can employ (R)EI to train any existed imaging networks to achieve fully unsupervised learning to image without changing the architectures. In addition to our demonstrated applications of REI (EI) on image inpainting, CT and MRI image reconstruction tasks, REI (EI) can be used to achieve new and fully unsupervised learning solutions to other inverse problems in computer vision and scientific imaging tasks, especially the cases when no groundtruth data is available for training.


<div align=center><img width="600" src="https://github.com/edongdongchen/REI/blob/main/images/fig1_mri.png"></div>

Figure 3: **Motivation.** The performance of EI degrades with increasing noise. From top to bottom: reconstructions of EI, supervised (Sup) baseline, and the proposed REI on 4× accelerated MRI with `Gaussian` noise level <img src="https://render.githubusercontent.com/render/math?math=\sigma"> = 0.01, 0.1, 0.2. PSNR values are shown in the top right corner of the images

![flexible](https://github.com/edongdongchen/REI/blob/main/images/fig_ct.png)
![flexible](https://github.com/edongdongchen/REI/blob/main/images/fig_ipt.png)
Figure 4: **More results.** From top to bottom: reconstruction of <img src="https://render.githubusercontent.com/render/math?math=A^{\dagger}y">, EI, REI, Sup and the groundtruth on the non-linear CT (with `MPG` noise) and Inpainting (with `Poisson` noise) tasks, respectively.

## [Frequently Asked Questions](/qa.md)
We collected some Frequently Asked Questions, please find the above Q & A.

## Run the code

1. Requirements: configure the environment by following: [environment.yml](https://github.com/edongdongchen/REI/blob/main/environment.yml) to run Inpainting and CT experiments. To run MRI experiments, please install the 'fastmri' package by `pip install fastmri`.

2. Find the implementation of Robust Equivariant Imaging (**REI**):
   * REI for the `accelerated MRI` task and the `Inpainting` task: [rei_end2end.py](https://github.com/edongdongchen/REI/blob/main/rei/closure/rei_end2end.py)
   * REI for the `low-dose and sparse-view CT` task: [rei_end2end_ct.py](https://github.com/edongdongchen/REI/blob/main/rei/closure/rei_end2end_ct.py)
   * Find our implementation of `SURE` for `Gaussian` and `Poisson` noise models at: [rei_end2end.py](https://github.com/edongdongchen/REI/blob/main/rei/closure/rei_end2end.py)
   * Find our implementation of `SURE` for `Mixed Poisson-Gaussian` noise model at: [rei_end2end_ct.py](https://github.com/edongdongchen/REI/blob/main/rei/closure/rei_end2end_ct.py)

3. Download datasets from the below source and move them under the folders: `./dataset/mri`, `./dataset/Urban100`, and `./dataset/CT`, repectively:
   * fastMRI (only the subset 'Knee MRI'): https://fastmri.med.nyu.edu/
   * Urban100: https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip
   * CT100: https://www.kaggle.com/kmader/siim-medical-images

4. **Train**: run the below scripts to train REI models:
   * run `./demo_scripts/demo_mri.py`, `./demo_scripts/demo_inpainting.py`, `./demo_scripts/demo_ct.py` to train REI for MRI, Inpainting, and CT tasks, respectively.
   * or run `train_bash.sh` to train REI models on all tasks.
   
   ```
   bash train_bash.sh
   ```

5. **Test**: run 'demo_test.py' to test the performance (PSNR) of a trained model on a specific task.
   ```
   python3 demo_test.py
   ``` 
   We also provide the trained models used in the paper which can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1xvxEt9o7PTk6ztiTFy1MGzlPjFTi9iDL?usp=sharing). Please put the downloaded folder 'ckp' in the root path. 
   

6. To solve a new inverse problem, one *only* needs to 
   * step 1: implement their own forward model (physics of sensing model)
   * step 2: determine the transformation group
   * step 3: specify the path of new dataset

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
