#!/usr/bin/env bash

#GPU 0
{ \
python3 demo_train.py --gpu=0 --task='mri' --noise-sigma=0.01; \
python3 demo_train.py --gpu=0 --task='mri' --noise-sigma=0.05; \
python3 demo_train.py --gpu=0 --task='mri' --noise-sigma=0.1; \
python3 demo_train.py --gpu=0 --task='mri' --noise-sigma=0.2;}&

#
#GPU 1
{ \
python3 demo_train.py --gpu=1 --task='inpainting' --noise-sigma=0.01; \
python3 demo_train.py --gpu=1 --task='inpainting' --noise-sigma=0.05; \
python3 demo_train.py --gpu=1 --task='inpainting' --noise-sigma=0.1;}&


#GPU 2
{ \
python3 demo_train.py --gpu=2 --task='ct';}